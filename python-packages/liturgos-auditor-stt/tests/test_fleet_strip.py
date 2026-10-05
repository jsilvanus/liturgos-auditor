"""AUDITOR_STT_STRIP=fleet: the audio of a source_url job is stripped by a fleet worker and PUT back."""

import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.jobs.fleet import FleetConfig, FleetStripper, strip_spec
from auditor_stt.serve.jobs.runner import JobRunner, RunnerConfig
from auditor_stt.serve.jobs.store import JobStore
from auditor_stt.serve.queue import InferenceQueue

from test_jobs_api import FIVE, _copy_normalize, _file_server, _Host, _no_gap, _until, _wait_done, _wav_bytes

PUBLIC = "http://auditor.test"
SOURCE = "http://127.0.0.1:9/bucket/talk.mp4?X-Amz-Signature=sekret"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("AUDITOR_STT_"):
            monkeypatch.delenv(name, raising=False)


class FakeFleet:
    """An fffleet orchestrator in miniature. `behaviour` decides what the 'worker' does with a submitted job."""

    def __init__(self, behaviour="deliver", wav=b""):
        self.behaviour = behaviour
        self.wav = wav
        self.jobs = {}
        self.requests = []
        self.svc = None
        self.delivered = threading.Event()

    def handler(self, request: httpx.Request):
        path = request.url.path
        self.requests.append((request.method, path, request.headers.get("authorization")))
        if path == "/v1/auth/token":
            return httpx.Response(200, json={"access_token": "tok-1", "token_type": "Bearer", "expires_in": 3600})
        if self.behaviour == "down":
            return httpx.Response(503, json={"error": {"code": "UNAVAILABLE", "message": "no"}})
        if path == "/v1/jobs" and request.method == "POST":
            spec = json.loads(request.content)
            if self.behaviour == "reject":
                return httpx.Response(422, json={"error": {"message": f"bad input {spec['inputs'][0]['uri']}"}})
            self.jobs[spec["id"]] = {"spec": spec, "state": "running", "error": None}
            threading.Thread(target=self._work, args=(spec["id"],), daemon=True).start()
            return httpx.Response(201, json={"id": spec["id"], "state": "running"})
        fleet_id = path.rsplit("/", 1)[-1]
        job = self.jobs.get(fleet_id)
        if job is None:
            return httpx.Response(404, json={"error": {"message": "unknown"}})
        if request.method == "DELETE":
            job["state"] = "cancelled"
            return httpx.Response(202, json={"id": fleet_id, "state": "cancelled"})
        return httpx.Response(200, json={"id": fleet_id, "state": job["state"], "error": job["error"]})

    def _work(self, fleet_id):
        job = self.jobs[fleet_id]
        if self.behaviour == "hang":
            return
        if self.behaviour == "fail":
            job["state"], job["error"] = "failed", {"code": "INPUT_FAILED", "message": f"could not fetch {job['spec']['inputs'][0]['uri']}"}
            return
        if self.behaviour == "deliver":
            target = job["spec"]["outputs"][0]["uri"]
            parts = urlsplit(target)
            response = self.svc.client.put(f"{parts.path}?{parts.query}", content=self.wav)
            assert response.status_code == 204, response.text
            self.delivered.set()
        job["state"] = "succeeded"


@contextmanager
def _service(tmp_path, fleet, *, fallback=True, normalize=_copy_normalize, config=None):
    host = _Host()
    queue = InferenceQueue()
    store = JobStore(tmp_path / "data" / "jobs")
    fleet_config = config or FleetConfig(
        mode="fleet", url="http://fleet.test", token="static", public_url=PUBLIC, fallback=fallback, poll_seconds=0.01
    )
    stripper = FleetStripper(fleet_config, client=httpx.Client(transport=httpx.MockTransport(fleet.handler)))
    runner = JobRunner(
        store,
        lambda: host,
        queue,
        RunnerConfig(retry_backoff=(0.0, 0.0), model_poll_seconds=0.01),
        normalize=normalize,
        find_gap=_no_gap,
        fleet=stripper,
        public_url=PUBLIC,
        fleet_fallback=fallback,
        fetch_limit=50 * 1024 * 1024,
        fetch_timeout=5.0,
    )
    app = create_app(model_host=host, queue=queue, data_dir=tmp_path / "data", job_store=store, runner=runner)
    app.state.source_url_hosts = ["127.0.0.1"]
    with TestClient(app) as client:
        svc = SimpleNamespace(client=client, store=store, host=host, runner=runner, app=app)
        fleet.svc = svc
        yield svc


def _submit(svc, url=SOURCE, **fields):
    return svc.client.post("/v1/jobs", data={"source_url": url, **FIVE, **fields})


def test_fleet_strips_and_only_the_audio_reaches_the_service(tmp_path):
    fleet = FakeFleet(wav=_wav_bytes(tmp_path, 30))
    with _service(tmp_path, fleet) as svc:
        response = _submit(svc)
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        job = _wait_done(svc, job_id)
        assert job["status"] == "completed", job
        assert job["chunks_total"] == 6

        spec = fleet.jobs[f"auditor-strip-{job_id}"]["spec"]
        assert spec["kind"] == "batch"
        assert spec["inputs"] == [{"name": "src", "uri": SOURCE}]
        output = spec["outputs"][0]
        assert output["uri"].startswith(f"{PUBLIC}/v1/jobs/{job_id}/audio?token=")
        assert "-vn" in spec["ffmpeg"]["args"] and "{{input:src}}" in spec["ffmpeg"]["args"]
        assert ("POST", "/v1/jobs", "Bearer static") in fleet.requests

        # The URL and the ingest token are not kept once the audio has arrived, and nothing was downloaded here.
        manifest = svc.store.load(job_id)
        assert manifest["source"] == {"kind": "url", "path": None}
        assert not list(svc.store.source_dir(job_id).glob("*"))


def test_unreachable_fleet_falls_back_to_a_local_strip(tmp_path):
    body = _wav_bytes(tmp_path, 30)
    fleet = FakeFleet("down")
    with _file_server({"/bucket/talk.mp4": (200, body, {})}) as base, _service(tmp_path, fleet) as svc:
        job_id = _submit(svc, f"{base}/bucket/talk.mp4?sig=1").json()["id"]
        job = _wait_done(svc, job_id)
        assert job["status"] == "completed", job
        assert job["chunks_total"] == 6


def test_without_fallback_an_unreachable_fleet_fails_the_job(tmp_path):
    fleet = FakeFleet("down")
    with _service(tmp_path, fleet, fallback=False) as svc:
        job = _wait_done(svc, _submit(svc).json()["id"])
        assert job["status"] == "failed"
        assert "fleet is unavailable" in job["error"]


def test_a_failed_fleet_job_fails_the_job_without_leaking_the_url(tmp_path):
    fleet = FakeFleet("fail")
    with _service(tmp_path, fleet) as svc:
        job = _wait_done(svc, _submit(svc).json()["id"])
        assert job["status"] == "failed"
        assert "could not fetch" in job["error"]
        assert "sekret" not in job["error"] and "127.0.0.1" not in job["error"]


def test_a_rejected_spec_fails_the_job(tmp_path):
    fleet = FakeFleet("reject")
    with _service(tmp_path, fleet) as svc:
        job = _wait_done(svc, _submit(svc).json()["id"])
        assert job["status"] == "failed"
        assert "sekret" not in job["error"]


def test_cancel_stops_the_fleet_job(tmp_path):
    fleet = FakeFleet("hang")
    with _service(tmp_path, fleet) as svc:
        job_id = _submit(svc).json()["id"]
        _until(lambda: f"auditor-strip-{job_id}" in fleet.jobs)
        assert svc.client.delete(f"/v1/jobs/{job_id}").status_code in (200, 202, 204)
        _until(lambda: fleet.jobs[f"auditor-strip-{job_id}"]["state"] == "cancelled")


def test_client_credentials_log_in_once_and_use_the_bearer_token(tmp_path):
    fleet = FakeFleet(wav=_wav_bytes(tmp_path, 10))
    config = FleetConfig(
        mode="fleet", url="http://fleet.test", client_id="auditor", client_secret="shh", public_url=PUBLIC, poll_seconds=0.01
    )
    with _service(tmp_path, fleet, config=config) as svc:
        assert _wait_done(svc, _submit(svc).json()["id"])["status"] == "completed"
    logins = [r for r in fleet.requests if r[1] == "/v1/auth/token"]
    assert len(logins) == 1
    assert ("POST", "/v1/jobs", "Bearer tok-1") in fleet.requests


# --- the audio endpoint ------------------------------------------------------


def _pending_job(svc):
    """A queued url job whose runner has not started (so the endpoint can be probed directly)."""
    manifest = svc.store.create(
        params={}, source={"kind": "url", "path": None, "url": SOURCE, "ingest_token": "tok-abc"}
    )
    return manifest["id"]


def test_audio_endpoint_checks_token_size_and_format(tmp_path):
    fleet = FakeFleet("hang")
    with _service(tmp_path, fleet) as svc:
        job_id = _pending_job(svc)
        wav = _wav_bytes(tmp_path, 5)
        url = f"/v1/jobs/{job_id}/audio"
        assert svc.client.put(f"{url}?token=nope", content=wav).status_code == 403
        assert svc.client.put(url, content=wav).status_code == 403
        assert svc.client.put(f"{url}?token=tok-abc", content=b"this is not a wav").status_code == 422
        svc.app.state.max_ingest_bytes = 1000
        assert svc.client.put(f"{url}?token=tok-abc", content=wav).status_code == 413
        assert not list(svc.store.job_dir(job_id).glob("audio.wav*"))
        svc.app.state.max_ingest_bytes = 10**9
        assert svc.client.put(f"{url}?token=tok-abc", content=wav).status_code == 204
        assert svc.store.audio_path(job_id).is_file()
        assert svc.client.put("/v1/jobs/" + "0" * 32 + "/audio?token=x", content=wav).status_code == 404


def test_audio_endpoint_refuses_jobs_that_are_not_waiting(tmp_path):
    fleet = FakeFleet("hang")
    with _service(tmp_path, fleet) as svc:
        job_id = _pending_job(svc)
        svc.store.set_status(job_id, "failed", error="x")
        response = svc.client.put(f"/v1/jobs/{job_id}/audio?token=tok-abc", content=_wav_bytes(tmp_path, 5))
        assert response.status_code == 409
        upload = svc.store.create(params={}, source={"kind": "upload", "path": None})["id"]
        assert svc.client.put(f"/v1/jobs/{upload}/audio?token=", content=_wav_bytes(tmp_path, 5)).status_code == 403


def test_audio_endpoint_needs_no_api_key_but_job_routes_still_do(tmp_path):
    fleet = FakeFleet("hang")
    with _service(tmp_path, fleet) as svc:
        svc.app.state.api_key = "k"
        job_id = _pending_job(svc)
        assert svc.client.get(f"/v1/jobs/{job_id}").status_code == 401
        assert svc.client.put(f"/v1/jobs/{job_id}/audio?token=tok-abc", content=_wav_bytes(tmp_path, 5)).status_code == 204


# --- configuration ------------------------------------------------------------


def test_config_defaults_to_local_and_validates_fleet_mode():
    assert not FleetConfig.from_env({}).enabled
    with pytest.raises(ValueError, match="AUDITOR_STT_STRIP"):
        FleetConfig.from_env({"AUDITOR_STT_STRIP": "cloud"})
    with pytest.raises(ValueError, match="AUDITOR_STT_PUBLIC_URL"):
        FleetConfig.from_env({"AUDITOR_STT_STRIP": "fleet", "AUDITOR_STT_FLEET_URL": "http://f"})
    config = FleetConfig.from_env(
        {
            "AUDITOR_STT_STRIP": "fleet",
            "AUDITOR_STT_FLEET_URL": "http://f/",
            "AUDITOR_STT_PUBLIC_URL": "https://a.example/",
            "AUDITOR_STT_FLEET_FALLBACK": "off",
        }
    )
    assert config.enabled and config.url == "http://f" and config.public_url == "https://a.example" and not config.fallback


def test_strip_spec_is_a_valid_batch_job():
    spec = strip_spec("abc", "https://h/v.mp4", "https://a/v1/jobs/abc/audio?token=t")
    assert spec["id"] == "auditor-strip-abc" and spec["kind"] == "batch"
    assert spec["ffmpeg"]["args"][-1] == "{{output:wav}}"


def test_without_fleet_a_source_url_is_still_fetched_here(tmp_path):
    body = _wav_bytes(tmp_path, 30)
    fleet = FakeFleet()
    with _file_server({"/talk.wav": (200, body, {})}) as base, _service(tmp_path, fleet) as svc:
        svc.runner._fleet = None  # local mode
        job = _wait_done(svc, _submit(svc, f"{base}/talk.wav").json()["id"])
        assert job["status"] == "completed"
        assert fleet.requests == []
