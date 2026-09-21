from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.queue import InferenceQueue


class _NotLoadedHost:
    model_id = "stub-model"
    device = None
    compute_type = None
    loaded = False

    def load(self):
        pass


class _LoadedHost(_NotLoadedHost):
    device = "cpu"
    compute_type = "int8"
    loaded = True


def test_status_reports_queue_and_model():
    app = create_app(model_host=_LoadedHost(), queue=InferenceQueue(max_queue=5))
    with TestClient(app) as client:
        resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json() == {
        "queue": {"live_depth": 0, "batch_depth": 0, "max_queue": 5},
        "model": {
            "status": "ok",
            "model_id": "stub-model",
            "device": "cpu",
            "compute_type": "int8",
            "loaded": True,
        },
    }


def test_status_passes_the_live_queue_stats_through():
    class _BusyQueue(InferenceQueue):
        def stats(self):
            return {"live_depth": 2, "batch_depth": 7, "max_queue": self.max_queue}

    app = create_app(model_host=_LoadedHost(), queue=_BusyQueue(max_queue=8))
    with TestClient(app) as client:
        assert client.get("/status").json()["queue"] == {"live_depth": 2, "batch_depth": 7, "max_queue": 8}


def test_status_model_section_matches_the_health_body():
    app = create_app(model_host=_LoadedHost(), queue=InferenceQueue(max_queue=8))
    with TestClient(app) as client:
        assert client.get("/status").json()["model"] == client.get("/health").json()


def test_status_is_200_even_while_the_model_is_loading():
    # /health is the readiness probe (503 until loaded); /status is for inspection.
    app = create_app(model_host=_NotLoadedHost(), queue=InferenceQueue(max_queue=8))
    with TestClient(app) as client:
        resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["model"]["loaded"] is False
    assert resp.json()["model"]["status"] == "loading"


def test_health_body_is_unchanged_by_status():
    app = create_app(model_host=_LoadedHost(), queue=InferenceQueue(max_queue=8))
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model_id": "stub-model",
            "device": "cpu",
            "compute_type": "int8",
            "loaded": True,
        }
