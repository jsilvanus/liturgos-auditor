"""Queue estimates, the duration probe, the job rate and the /live and /ready probes."""

import os
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.jobs.estimate import job_duration, queue_overview
from auditor_stt.serve.jobs.fleet import FleetConfig, FleetStripper
from auditor_stt.serve.jobs.probe import ProbeError, parse_duration, probe_local
from auditor_stt.serve.jobs.rate import RateTracker
from auditor_stt.serve.jobs.runner import JobRunner, RunnerConfig
from auditor_stt.serve.jobs.store import JobStore
from auditor_stt.serve.queue import InferenceQueue

from test_fleet_strip import FakeFleet, SOURCE
from test_jobs_api import _Host, _copy_normalize, _no_gap, _service, _submit_upload, _until, _wait_done, _wav_bytes, _job
from test_fleet_strip import _service as _fleet_service
from test_fleet_strip import _submit as _fleet_submit


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("AUDITOR_STT_"):
            monkeypatch.delenv(name, raising=False)


def _manifest(job_id, status, *, duration=None, current=0.0):
    return {"id": job_id, "status": status, "duration_seconds": duration, "current_seconds": current, "audio": None}


# --- the estimate ----------------------------------------------------------------


def test_job_duration_prefers_the_measured_audio():
    assert job_duration({"audio": {"duration_seconds": 61.5}, "duration_seconds": 60.0}) == 61.5
    assert job_duration({"audio": None, "duration_seconds": 60.0}) == 60.0
    assert job_duration({"audio": None, "duration_seconds": None}) is None


def test_estimate_adds_the_rest_of_the_running_job_and_older_queued_jobs():
    manifests = [
        _manifest("a", "running", duration=100.0, current=40.0),
        _manifest("b", "queued", duration=50.0),
        _manifest("c", "queued", duration=20.0),
    ]
    overview = queue_overview(manifests, rate=2.0)
    assert overview["jobs"]["b"] == {
        "position": 1, "seconds_ahead": 60.0, "unknown_ahead": 0,
        "estimated_start_seconds": 30.0, "estimated_finish_seconds": 55.0, "estimate_is_minimum": False,
    }
    assert overview["jobs"]["c"]["seconds_ahead"] == 110.0
    assert overview["jobs"]["c"]["position"] == 2
    assert overview["jobs"]["c"]["estimated_start_seconds"] == 55.0
    assert overview["summary"] == {"queued": 2, "running": 1, "queued_seconds": 70.0, "unknown_duration": 0}


def test_an_unknown_duration_ahead_leaves_the_estimate_unknown():
    manifests = [_manifest("a", "queued", duration=None), _manifest("b", "queued", duration=10.0)]
    overview = queue_overview(manifests, rate=1.0)
    assert overview["jobs"]["a"]["estimated_start_seconds"] == 0.0
    assert overview["jobs"]["a"]["estimated_finish_seconds"] is None
    assert overview["jobs"]["b"]["unknown_ahead"] == 1
    assert overview["jobs"]["b"]["estimated_start_seconds"] is None
    assert overview["summary"]["unknown_duration"] == 1


def test_without_a_rate_and_while_live_runs():
    manifests = [_manifest("a", "queued", duration=10.0)]
    assert queue_overview(manifests, rate=None)["jobs"]["a"]["estimated_start_seconds"] is None
    block = queue_overview(manifests, rate=1.0, live_active=True)["jobs"]["a"]
    assert block["estimate_is_minimum"] is True


# --- the rate --------------------------------------------------------------------


def test_rate_is_the_median_of_the_last_jobs_and_survives_a_restart(tmp_path):
    path = tmp_path / "rate.json"
    tracker = RateTracker(path, keep=3)
    assert tracker.rate() is None
    for audio, wall in [(100, 100), (100, 50), (100, 25), (100, 20)]:  # rates 1, 2, 4, 5
        tracker.record(audio, wall)
    assert tracker.sample_count() == 3
    assert tracker.rate() == 4.0  # median of 2, 4, 5
    assert RateTracker(path, keep=3).rate() == 4.0


def test_rate_ignores_nonsense(tmp_path):
    tracker = RateTracker(tmp_path / "rate.json")
    for audio, wall in [(0, 5), (5, 0), (-1, 5), (True, 5), (None, 5)]:
        tracker.record(audio, wall)
    assert tracker.rate() is None
    (tmp_path / "rate.json").write_text("not json")
    assert RateTracker(tmp_path / "rate.json").rate() is None


# --- probing ---------------------------------------------------------------------


def test_parse_duration():
    assert parse_duration("12.5") == 12.5
    for bad in ("", "N/A", "nan", "inf", "0", "-3", None):
        with pytest.raises(ProbeError):
            parse_duration(bad)


def test_probe_local_reads_a_wav(tmp_path):
    path = tmp_path / "talk.wav"
    path.write_bytes(_wav_bytes(tmp_path, 12))
    assert probe_local(path, 10) == pytest.approx(12.0, abs=0.1)
    (tmp_path / "junk.bin").write_bytes(b"not media")
    with pytest.raises(ProbeError):
        probe_local(tmp_path / "junk.bin", 10)


# --- through the API -------------------------------------------------------------


class _Gate(_Host):
    """Holds the first chunk so that later jobs stay queued."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.started = threading.Event()
        self.hook = self._wait

    def _wait(self, _offset):
        self.started.set()
        assert self.release.wait(20)


def test_queued_jobs_show_position_and_estimate(tmp_path):
    host = _Gate()
    with _service(tmp_path, host=host) as svc:
        first = _submit_upload(svc, tmp_path, seconds=30)
        assert first.status_code == 202
        assert host.started.wait(10)
        second = _submit_upload(svc, tmp_path, seconds=20, duration_seconds="20")
        third = _submit_upload(svc, tmp_path, seconds=10)
        svc.runner.rate.record(30, 10)  # 3 audio seconds per second

        one = _job(svc, second.json()["id"])
        assert one["status"] == "queued"
        assert one["duration_seconds"] == 20.0
        assert svc.store.load(one["id"])["duration_source"] == "client"
        assert one["queue"] == {
            "position": 1, "seconds_ahead": 30.0, "unknown_ahead": 0,
            "estimated_start_seconds": 10.0, "estimated_finish_seconds": pytest.approx(16.7, abs=0.05),
            "estimate_is_minimum": False,
        }
        two = _job(svc, third.json()["id"])
        assert two["queue"]["position"] == 2
        assert two["queue"]["seconds_ahead"] == 50.0
        assert two["duration_seconds"] == pytest.approx(10.0, abs=0.1)  # probed, nobody said
        assert svc.store.load(two["id"])["duration_source"] == "probe"

        running = _job(svc, first.json()["id"])
        assert running["queue"] is None

        listed = {job["id"]: job for job in svc.client.get("/v1/jobs").json()["jobs"]}
        assert listed[one["id"]]["queue"]["position"] == 1
        assert listed[first.json()["id"]]["queue"] is None

        status = svc.client.get("/status").json()
        assert status["jobs"] == {"queued": 2, "running": 1, "queued_seconds": pytest.approx(30.0, abs=0.1), "unknown_duration": 0}
        assert status["rate"] == 3.0
        assert status["rate_samples"] == 1
        assert status["live"] is None

        host.release.set()
        assert _wait_done(svc, first.json()["id"])["status"] == "completed"
        assert _wait_done(svc, third.json()["id"])["status"] == "completed"
        assert _job(svc, third.json()["id"])["queue"] is None


def test_an_unprobeable_upload_stays_unknown(tmp_path):
    host = _Gate()
    with _service(tmp_path, host=host) as svc:
        _submit_upload(svc, tmp_path, seconds=30)
        assert host.started.wait(10)
        junk = svc.client.post("/v1/jobs", files={"file": ("x.bin", b"not media", "application/octet-stream")}, data={})
        job = _job(svc, junk.json()["id"])
        assert job["duration_seconds"] is None
        assert svc.store.load(job["id"])["duration_source"] is None
        assert job["queue"]["estimated_finish_seconds"] is None
        assert svc.client.get("/status").json()["jobs"]["unknown_duration"] == 1
        host.release.set()


def test_bad_duration_is_refused(tmp_path):
    with _service(tmp_path) as svc:
        for bad in ("0", "-5", "nan", "abc"):
            assert _submit_upload(svc, tmp_path, duration_seconds=bad).status_code == 422


def test_a_finished_job_records_its_duration_and_rate(tmp_path):
    with _service(tmp_path) as svc:
        job_id = _submit_upload(svc, tmp_path, seconds=30).json()["id"]
        assert _wait_done(svc, job_id)["status"] == "completed"
        manifest = svc.store.load(job_id)
        assert manifest["duration_source"] == "normalised"
        assert manifest["duration_seconds"] == pytest.approx(30.0)
        _until(lambda: svc.runner.rate.sample_count() == 1)
        assert svc.runner.rate.rate() > 0


def test_status_reports_live_sessions(tmp_path):
    class _Manager:
        config = type("C", (), {"max_sessions": 2})()
        sessions = {}

        def active_sessions(self):
            return [object()]

        async def shutdown(self):
            pass

    with _service(tmp_path) as svc:
        svc.app.state.live = _Manager()
        assert svc.client.get("/status").json()["live"] == {"active": True, "sessions": 1, "max_sessions": 2}


def test_status_stays_behind_the_key_but_probes_are_open(tmp_path):
    with _service(tmp_path, api_key="k") as svc:
        assert svc.client.get("/status").status_code == 401
        for path in ("/health", "/live", "/ready"):
            assert svc.client.get(path).status_code == 200, path


# --- the fleet probe -------------------------------------------------------------


def test_a_url_job_is_probed_by_the_fleet(tmp_path):
    fleet = FakeFleet(wav=_wav_bytes(tmp_path, 30))
    fleet.probe_seconds = 1234.5
    with _fleet_service(tmp_path, fleet) as svc:
        job_id = _fleet_submit(svc).json()["id"]
        probe = fleet.jobs[f"auditor-probe-{job_id}"]["spec"]
        assert probe["type"] == "ffprobe"
        assert probe["inputs"] == [{"name": "src", "uri": SOURCE}]
        assert probe["outputs"] == []
        assert probe["timeoutMs"] == 10000
        manifest = svc.store.load(job_id)
        assert manifest["duration_source"] in ("probe", "normalised")
        _wait_done(svc, job_id)


def test_a_failed_fleet_probe_is_not_an_error(tmp_path):
    fleet = FakeFleet("reject")
    with _fleet_service(tmp_path, fleet) as svc:
        response = _fleet_submit(svc)
        assert response.status_code == 202
        job = _wait_done(svc, response.json()["id"])
        assert job["status"] == "failed"  # the strip is rejected too; the point is that submit still answered 202
        assert job["duration_seconds"] is None


def test_an_unreachable_fleet_does_not_block_submit(tmp_path):
    fleet = FakeFleet("down")
    with _fleet_service(tmp_path, fleet) as svc:
        assert _fleet_submit(svc).status_code == 202


# --- liveness and readiness ------------------------------------------------------


class _Unloaded:
    model_id = "stub-model"
    device = None
    compute_type = None
    loaded = False

    def load(self):
        pass


class _Loaded(_Unloaded):
    device = "cpu"
    compute_type = "int8"
    loaded = True


class _Failing(_Unloaded):
    def load(self):
        from auditor_stt.serve.model import ModelLoadError

        raise ModelLoadError("Could not load model '/secret/path/model' on any device")


def test_live_is_200_whatever_the_model_does():
    for host in (_Unloaded(), _Loaded(), _Failing()):
        with TestClient(create_app(model_host=host, queue=InferenceQueue())) as client:
            assert client.get("/live").json() == {"status": "ok"}


def test_ready_follows_the_model():
    with TestClient(create_app(model_host=_Loaded(), queue=InferenceQueue())) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["state"] == "ready"
        assert response.json()["error"] is None

    with TestClient(create_app(model_host=_Unloaded(), queue=InferenceQueue())) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["state"] == "loading"


def test_ready_tells_a_failed_load_from_a_slow_one_without_paths():
    with TestClient(create_app(model_host=_Failing(), queue=InferenceQueue())) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["state"] == "failed"
        assert body["error"] == "The model failed to load"
        assert "secret" not in response.text
        assert client.get("/health").status_code == 503  # /health keeps its old meaning


def test_ready_needs_a_writable_job_store(tmp_path, monkeypatch):
    with _service(tmp_path) as svc:
        assert svc.client.get("/ready").status_code == 200
        monkeypatch.setattr(os, "access", lambda *_args, **_kw: False)
        response = svc.client.get("/ready")
        assert response.status_code == 503
        assert response.json()["state"] == "storage_unwritable"
