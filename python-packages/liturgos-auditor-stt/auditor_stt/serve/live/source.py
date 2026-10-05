"""PCM sources for live sessions: an fffleet stream job, or a local ffmpeg.

Both give 16 kHz mono signed 16-bit little-endian PCM through `read(n)` (bytes,
empty at the end) and `close()`. `close()` is safe to call from another thread.
"""

import logging
import shutil
import subprocess
import threading

import httpx

from ..jobs.fleet import FleetStripper, FleetUnavailableError, _error_text, _scrubber

logger = logging.getLogger(__name__)

LIVE_SCHEMES = ("rtsp", "rtsps", "srt")


def ffmpeg_args(url):
    """ffmpeg arguments after `-i <url>`'s own options; `url` is replaced by the caller's placeholder."""
    return ["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]


def _input_options(url):
    return ["-rtsp_transport", "tcp"] if url.lower().startswith(("rtsp:", "rtsps:")) else []


def stream_spec(fleet_id, url, requires=()):
    """The fleet job: ffmpeg pulls `url` and writes PCM to stdout, which this service reads from the fleet."""
    return {
        "contract": 1,
        "id": fleet_id,
        "kind": "stream",
        "stdout": True,
        "labels": {"app": "auditor-stt", "role": "live"},
        "requires": list(requires),
        "inputs": [{"name": "src", "uri": url}],
        "ffmpeg": {"args": [*_input_options(url), "-i", "{{input:src}}", *ffmpeg_args(url)]},
    }


class FleetPcmStream:
    """Stdout of a fleet stream job. Opening submits the job; closing cancels it."""

    def __init__(self, fleet: FleetStripper, fleet_id, url, *, requires=(), read_timeout=120.0):
        self.fleet = fleet
        self.fleet_id = fleet_id
        self._scrub = _scrubber(url)
        self._closed = threading.Event()
        self._response = None
        self._iter = None
        self._rest = b""
        self._submit(stream_spec(fleet_id, url, requires))
        self._read_timeout = read_timeout
        self._connect()

    def _submit(self, spec):
        response = self.fleet._request("POST", "/v1/jobs", json=spec)
        if response.status_code in (401, 403) or response.status_code >= 500:
            raise FleetUnavailableError(f"fleet answered {response.status_code}")
        if response.status_code == 422:
            raise OSError(self._scrub(_error_text(response)) or "the fleet rejected the live job")
        if response.status_code not in (200, 201, 202):
            raise FleetUnavailableError(f"fleet answered {response.status_code}")

    def _connect(self):
        # The orchestrator holds the request while the job is queued.
        client = self.fleet._client
        request = client.build_request(
            "GET",
            f"{self.fleet.config.url}/v1/jobs/{self.fleet_id}/stdout",
            headers=self.fleet._auth_headers(),
            timeout=httpx.Timeout(15.0, read=self._read_timeout),
        )
        try:
            self._response = client.send(request, stream=True)
        except httpx.HTTPError as exc:
            self.close()
            raise OSError(f"fleet stdout unreachable ({type(exc).__name__})") from None
        if self._response.status_code != 200:
            status = self._response.status_code
            self.close()
            raise OSError(f"fleet stdout answered {status}")
        self._iter = self._response.iter_raw(65536)

    def read(self, size):
        while len(self._rest) < size and not self._closed.is_set():
            try:
                self._rest += next(self._iter)
            except StopIteration:
                break
            except (httpx.HTTPError, OSError, ValueError):
                break
        data, self._rest = self._rest[:size], self._rest[size:]
        return data

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            if self._response is not None:
                self._response.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.fleet._request("DELETE", f"/v1/jobs/{self.fleet_id}")
        except FleetUnavailableError:
            logger.warning("Could not cancel fleet live job %s", self.fleet_id)


class LocalPcmStream:
    """ffmpeg run in this process; the fallback when no fleet is configured or reachable."""

    def __init__(self, url, *, ffmpeg="ffmpeg"):
        if shutil.which(ffmpeg) is None:
            raise OSError("ffmpeg is not installed")
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", *_input_options(url), "-i", url, *ffmpeg_args(url)]
        self._scrub = _scrubber(url)
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

    def read(self, size):
        data = b""
        while len(data) < size:
            chunk = self._process.stdout.read(size - len(data))
            if not chunk:
                break
            data += chunk
        return data

    def close(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        try:
            self._process.stdout.close()
        except Exception:  # noqa: BLE001
            pass
