import logging

import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.auth import require_api_key
from auditor_stt.serve.queue import InferenceQueue

KEY = "s3cret-key"
PROTECTED = ["/inference", "/status", "/v1/audio/transcriptions"]


class _StubHost:
    model_id = "stub-model"
    device = "cpu"
    compute_type = "int8"
    loaded = True

    def load(self):
        pass

    def transcribe(self, audio_path, language=None, **options):
        return {"text": "moi maailma", "language": language, "segments": []}


def _app(**kwargs):
    return create_app(model_host=_StubHost(), queue=InferenceQueue(max_queue=8), **kwargs)


def _call(client, path, **kwargs):
    if path == "/status":
        return client.get(path, **kwargs)
    return client.post(path, files={"file": ("clip.wav", b"x", "audio/wav")}, **kwargs)


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch):
    monkeypatch.delenv("AUDITOR_STT_API_KEY", raising=False)


def test_health_stays_open_when_a_key_is_configured():
    with TestClient(_app(api_key=KEY)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 200


@pytest.mark.parametrize("path", PROTECTED)
def test_missing_key_is_rejected_with_401(path):
    with TestClient(_app(api_key=KEY)) as client:
        resp = _call(client, path)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert set(resp.json()) == {"detail"}


@pytest.mark.parametrize("path", PROTECTED)
@pytest.mark.parametrize(
    "header",
    ["Bearer wrong-key", f"Basic {KEY}", KEY, "Bearer", f"Bearer {KEY}x", f"Bearer {KEY[:-1]}"],
)
def test_wrong_credentials_are_rejected_with_401(path, header):
    with TestClient(_app(api_key=KEY)) as client:
        resp = _call(client, path, headers={"Authorization": header})
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("path", PROTECTED)
def test_correct_key_is_accepted(path):
    with TestClient(_app(api_key=KEY)) as client:
        resp = _call(client, path, headers={"Authorization": f"Bearer {KEY}"})
    assert resp.status_code == 200


def test_bearer_scheme_is_case_insensitive():
    with TestClient(_app(api_key=KEY)) as client:
        resp = client.get("/status", headers={"Authorization": f"bearer {KEY}"})
    assert resp.status_code == 200


def test_non_ascii_credentials_are_a_401_not_a_server_error():
    with TestClient(_app(api_key=KEY)) as client:
        resp = client.get("/status", headers={"Authorization": "Bearer äö".encode("latin-1")})
    assert resp.status_code == 401


@pytest.mark.parametrize("path", PROTECTED)
def test_no_auth_at_all_when_no_key_is_configured(path):
    with TestClient(_app()) as client:
        assert _call(client, path).status_code == 200
        assert _call(client, path, headers={"Authorization": "Bearer anything"}).status_code == 200


def test_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("AUDITOR_STT_API_KEY", "from-env")
    with TestClient(_app()) as client:
        assert client.get("/status").status_code == 401
        assert client.get("/status", headers={"Authorization": "Bearer from-env"}).status_code == 200


def test_key_parameter_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("AUDITOR_STT_API_KEY", "from-env")
    with TestClient(_app(api_key="from-param")) as client:
        assert client.get("/status", headers={"Authorization": "Bearer from-env"}).status_code == 401
        assert client.get("/status", headers={"Authorization": "Bearer from-param"}).status_code == 200


def test_empty_key_parameter_disables_auth_even_if_the_environment_sets_one(monkeypatch):
    monkeypatch.setenv("AUDITOR_STT_API_KEY", "from-env")
    with TestClient(_app(api_key="")) as client:
        assert client.get("/status").status_code == 200


def test_dependency_protects_routers_included_later():
    app = _app(api_key=KEY)
    router = APIRouter(prefix="/v1/jobs")

    @router.get("/ping")
    def ping():
        return {"ok": True}

    app.include_router(router, dependencies=[Depends(require_api_key)])

    with TestClient(app) as client:
        assert client.get("/v1/jobs/ping").status_code == 401
        assert client.get("/v1/jobs/ping", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_startup_logs_that_auth_is_enabled_but_never_the_key(caplog):
    caplog.set_level(logging.INFO, logger="auditor_stt.serve.app")
    with TestClient(_app(api_key=KEY)):
        pass
    assert "API key auth enabled" in caplog.text
    assert KEY not in caplog.text


def test_startup_logs_that_auth_is_disabled(caplog):
    caplog.set_level(logging.INFO, logger="auditor_stt.serve.app")
    with TestClient(_app()):
        pass
    assert "API key auth disabled" in caplog.text
