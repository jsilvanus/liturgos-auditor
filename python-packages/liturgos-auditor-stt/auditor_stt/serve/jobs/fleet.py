"""Stripping a job's audio on an fffleet worker instead of in this process.

With AUDITOR_STT_STRIP=fleet, a job submitted with `source_url` is not downloaded
here. The runner submits an fffleet batch job whose input is that URL and whose
output is an HTTP PUT to this service (PUT /v1/jobs/{id}/audio, see ingest.py);
the worker fetches the file, runs ffmpeg and uploads the 16 kHz mono WAV. Only
that WAV (about 2 MB per minute) ever reaches this service.

This is a small client of fffleet's job contract v1 (POST /v1/jobs, GET/DELETE
/v1/jobs/{id}), written against the HTTP API so no Node code is needed here. The
fleet job id is derived from the job id, so submitting again after a restart finds
the same fleet job instead of starting a second one.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass

import httpx

from ..audio import NormalizationCancelledError

logger = logging.getLogger(__name__)

_ERROR_LIMIT = 300
_FINAL = ("succeeded", "failed", "cancelled")


class FleetUnavailableError(Exception):
    """The fleet could not be reached or did not accept the job; the caller may fall back to local work."""


class FleetJobError(Exception):
    """The fleet ran the job and it failed; the message is safe to show to a client."""


@dataclass
class FleetConfig:
    mode: str = "local"  # "local" | "fleet"
    url: str = ""
    token: str = ""
    client_id: str = ""
    client_secret: str = ""
    public_url: str = ""  # how fleet workers reach this service (for the audio PUT)
    fallback: bool = True  # strip locally when the fleet is unreachable
    poll_seconds: float = 2.0
    timeout_seconds: float = 6 * 3600.0
    request_timeout: float = 15.0
    max_poll_failures: int = 5

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env

        def text(name):
            return env.get(name, "").strip()

        mode = text("AUDITOR_STT_STRIP").lower() or "local"
        if mode not in ("local", "fleet"):
            raise ValueError("AUDITOR_STT_STRIP must be 'local' or 'fleet'")
        fallback = text("AUDITOR_STT_FLEET_FALLBACK").lower() not in ("0", "false", "no", "off")
        config = cls(
            mode=mode,
            url=text("AUDITOR_STT_FLEET_URL").rstrip("/"),
            token=text("AUDITOR_STT_FLEET_TOKEN"),
            client_id=text("AUDITOR_STT_FLEET_CLIENT_ID"),
            client_secret=text("AUDITOR_STT_FLEET_CLIENT_SECRET"),
            public_url=text("AUDITOR_STT_PUBLIC_URL").rstrip("/"),
            fallback=fallback,
        )
        if env.get("AUDITOR_STT_FLEET_POLL_SECONDS"):
            config.poll_seconds = float(env["AUDITOR_STT_FLEET_POLL_SECONDS"])
        if env.get("AUDITOR_STT_FLEET_TIMEOUT_SECONDS"):
            config.timeout_seconds = float(env["AUDITOR_STT_FLEET_TIMEOUT_SECONDS"])
        if mode == "fleet" and not (config.url and config.public_url):
            raise ValueError("AUDITOR_STT_STRIP=fleet needs AUDITOR_STT_FLEET_URL and AUDITOR_STT_PUBLIC_URL")
        return config

    @property
    def enabled(self):
        return self.mode == "fleet"


def strip_spec(job_id, source_url, ingest_url):
    """The fleet job: ffmpeg decodes the fetched file to 16 kHz mono PCM16 WAV and PUTs it to `ingest_url`."""
    return {
        "contract": 1,
        "id": f"auditor-strip-{job_id}",
        "kind": "batch",
        "labels": {"app": "auditor-stt"},
        "inputs": [{"name": "src", "uri": source_url}],
        "outputs": [{"name": "wav", "uri": ingest_url, "contentType": "audio/wav"}],
        "ffmpeg": {"args": ["-i", "{{input:src}}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", "{{output:wav}}"]},
    }


class FleetStripper:
    def __init__(self, config, *, client=None):
        self.config = config
        self._client = client or httpx.Client(timeout=config.request_timeout)
        self._token = None
        self._token_expires = 0.0
        self._lock = threading.Lock()

    # --- auth ------------------------------------------------------------

    def _auth_headers(self, refresh=False):
        config = self.config
        if config.token:
            return {"Authorization": f"Bearer {config.token}"}
        if not (config.client_id and config.client_secret):
            return {}
        with self._lock:
            if refresh or self._token is None or time.monotonic() >= self._token_expires:
                response = self._client.post(
                    f"{config.url}/v1/auth/token",
                    data={"grant_type": "client_credentials", "scope": "jobs"},
                    auth=(config.client_id, config.client_secret),
                )
                if response.status_code != 200:
                    raise FleetUnavailableError(f"fleet login answered {response.status_code}")
                body = response.json()
                self._token = body["access_token"]
                self._token_expires = time.monotonic() + max(30.0, float(body.get("expires_in", 3600)) - 60.0)
            return {"Authorization": f"Bearer {self._token}"}

    def _request(self, method, path, **kwargs):
        url = f"{self.config.url}{path}"
        try:
            response = self._client.request(method, url, headers=self._auth_headers(), **kwargs)
            if response.status_code == 401 and not self.config.token:
                response = self._client.request(method, url, headers=self._auth_headers(refresh=True), **kwargs)
        except httpx.HTTPError as exc:
            raise FleetUnavailableError(f"fleet unreachable ({type(exc).__name__})") from None
        return response

    # --- one strip -------------------------------------------------------

    def strip(self, job_id, source_url, ingest_url, *, cancel=None, on_phase=None):
        """Run the strip job and return when the worker has uploaded the audio. Blocking."""
        spec = strip_spec(job_id, source_url, ingest_url)
        scrub = _scrubber(source_url, ingest_url)
        response = self._request("POST", "/v1/jobs", json=spec)
        if response.status_code in (401, 403) or response.status_code >= 500:
            raise FleetUnavailableError(f"fleet answered {response.status_code}")
        if response.status_code == 422:
            raise FleetJobError(scrub(_error_text(response)) or "the fleet rejected the job")
        if response.status_code not in (200, 201, 202):
            raise FleetUnavailableError(f"fleet answered {response.status_code}")
        fleet_id = spec["id"]

        deadline = time.monotonic() + self.config.timeout_seconds
        failures = 0
        last_state = None
        while True:
            if cancel is not None and cancel.is_set():
                self._cancel_quietly(fleet_id)
                raise NormalizationCancelledError()
            if time.monotonic() >= deadline:
                self._cancel_quietly(fleet_id)
                raise FleetJobError("the fleet did not finish stripping the audio in time")
            try:
                response = self._request("GET", f"/v1/jobs/{fleet_id}")
                if response.status_code != 200:
                    raise FleetUnavailableError(f"fleet answered {response.status_code}")
                job = response.json()
                failures = 0
            except (FleetUnavailableError, ValueError) as exc:
                failures += 1
                if failures >= self.config.max_poll_failures:
                    raise FleetUnavailableError(str(exc) or "fleet status unreadable") from None
                self._sleep(cancel)
                continue
            state = job.get("state")
            if state != last_state and on_phase is not None:
                on_phase(state)
            last_state = state
            if state == "succeeded":
                return
            if state in _FINAL:
                raise FleetJobError(scrub(_job_error(job)) or f"the fleet job ended as {state}")
            self._sleep(cancel)

    def _sleep(self, cancel):
        if cancel is not None:
            cancel.wait(self.config.poll_seconds)
        else:
            time.sleep(self.config.poll_seconds)

    def _cancel_quietly(self, fleet_id):
        try:
            self._request("DELETE", f"/v1/jobs/{fleet_id}")
        except FleetUnavailableError:
            logger.warning("Could not cancel fleet job %s", fleet_id)

    def close(self):
        self._client.close()


def _scrubber(*secrets):
    """Remove the URLs (which can carry signatures or tokens) from text that may reach a client."""

    def scrub(text):
        text = str(text or "")
        for secret in secrets:
            text = text.replace(secret, "<url>")
        return " ".join(text.split())[:_ERROR_LIMIT]

    return scrub


def _error_text(response):
    try:
        body = response.json()
    except ValueError:
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return error.get("message", "")
    return str(error or "")


def _job_error(job):
    error = job.get("error")
    if isinstance(error, dict):
        return error.get("message") or error.get("code") or ""
    return str(error or "")
