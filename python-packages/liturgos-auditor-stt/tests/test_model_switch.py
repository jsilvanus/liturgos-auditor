import gc
import json
import logging
import shutil
import threading
import time
import weakref

import numpy as np
import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import DEFAULT_MODEL, SWITCH_NEEDS_KEY, create_app
from auditor_stt.serve.jobs.assemble import assemble
from auditor_stt.serve.jobs.pipeline import transcribe_chunk
from auditor_stt.serve.jobs.runner import JobRunner, RunnerConfig
from auditor_stt.serve.jobs.store import JobStore
from auditor_stt.serve.jobs.wav import open_pcm16_mono, write_pcm16_wav
from auditor_stt.serve.model import ModelLoadError
from auditor_stt.serve.queue import InferenceQueue
from auditor_stt.serve.registry import promote, register

KEY = "s3cret-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
RATE = 16000
ENV_NAMES = (
    "AUDITOR_STT_API_KEY",
    "AUDITOR_STT_MODEL",
    "AUDITOR_STT_ALLOWED_MODELS",
    "AUDITOR_STT_REGISTRY_DIR",
    "AUDITOR_STT_DATA_DIR",
    "AUDITOR_STT_MEDIA_ROOT",
)


class _Host:
    """Stub ModelHost. `load_release` / `release` (Events) hold load() / transcribe() until set."""

    def __init__(self, model_id, *, text=None, device="cpu", compute_type="int8", load_error=None, preloaded=False):
        self.model_id = model_id
        self.text = text or f"from {model_id}"
        self.load_error = load_error
        self.load_release = None
        self.load_started = threading.Event()
        self.release = None
        self.entered = threading.Event()
        self.hook = None  # hook(offset) runs in transcribe_array, before the result is returned
        self._settings = (device, compute_type)
        self.device = self.compute_type = None
        self.loaded = False
        if preloaded:
            self.load()

    def load(self):
        self.load_started.set()
        if self.load_release is not None:
            self.load_release.wait(10)
        if self.load_error is not None:
            raise self.load_error
        self.device, self.compute_type = self._settings
        self.loaded = True

    def transcribe(self, audio_path, language=None, **options):
        self.entered.set()
        if self.release is not None:
            self.release.wait(10)
        return {"text": self.text, "language": language, "segments": []}

    def transcribe_array(self, samples, language=None, *, time_offset=0.0, **options):
        if self.hook is not None:
            self.hook(time_offset)
        end = time_offset + len(samples) / RATE
        text = f"{self.model_id} at {time_offset:g}"
        segment = {"start": time_offset, "end": end, "text": text, "words": []}
        return {"text": text, "language": language, "segments": [segment]}


class _Factory:
    """host_factory stub: hands out the given hosts in order and remembers what it was asked for."""

    def __init__(self, *hosts):
        self.hosts = list(hosts)
        self.requested = []

    def __call__(self, model_id):
        self.requested.append(model_id)
        return self.hosts.pop(0)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def registry_dir(tmp_path):
    """A registry with whisper-fi v1 and v2 (v2 promoted)."""
    reg = tmp_path / "registry"
    for version in ("v1", "v2"):
        export = tmp_path / f"export-{version}"
        export.mkdir()
        (export / "model.bin").write_bytes(b"weights")
        (export / "gate.json").write_text(json.dumps({"passed": True, "dataset_version": "ds"}), encoding="utf-8")
        register(reg, "whisper-fi", version, export)
    promote(reg, "whisper-fi", "v2")
    return reg


def _app(old, factory, registry_dir, *, api_key=KEY, allowed=(), **kwargs):
    return create_app(
        model_host=old,
        queue=InferenceQueue(),
        api_key=api_key,
        host_factory=factory,
        registry_dir=registry_dir,
        allowed_models=allowed,
        **kwargs,
    )


def _switch(client, spec, headers=AUTH):
    return client.post("/model", json={"model": spec}, headers=headers)


def _transcribe(client, headers=AUTH):
    return client.post("/inference", files={"file": ("clip.wav", b"x", "audio/wav")}, headers=headers)


def _until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.005)


def _in_thread(fn):
    """Run `fn` in a thread; returns (thread, result dict whose "value" is set when it finishes)."""
    result = {}
    thread = threading.Thread(target=lambda: result.update(value=fn()))
    thread.start()
    return thread, result


# --- GET /model ---------------------------------------------------------------------------


def test_get_model_reports_the_current_model_and_the_registry(tmp_path, registry_dir):
    app = _app(_Host("stub", preloaded=True), _Factory(), registry_dir)
    with TestClient(app) as client:
        body = client.get("/model", headers=AUTH).json()

    assert body == {
        "model": {"model_id": "stub", "device": "cpu", "compute_type": "int8", "loaded": True, "source": "stub"},
        "registry": [{"name": "whisper-fi", "current_version": "v2", "versions": ["v1", "v2"]}],
    }


def test_get_model_needs_the_key_when_one_is_configured(registry_dir):
    app = _app(_Host("stub", preloaded=True), _Factory(), registry_dir)
    with TestClient(app) as client:
        assert client.get("/model").status_code == 401
        assert client.get("/model", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/model", headers=AUTH).status_code == 200


def test_get_model_is_readable_without_a_key_when_none_is_configured(registry_dir):
    app = _app(_Host("stub", preloaded=True), _Factory(), registry_dir, api_key=None)
    with TestClient(app) as client:
        assert client.get("/model").status_code == 200


def test_get_model_with_an_empty_registry(tmp_path):
    app = _app(_Host("stub", preloaded=True), _Factory(), tmp_path / "none")
    with TestClient(app) as client:
        assert client.get("/model", headers=AUTH).json()["registry"] == []


def test_the_registry_dir_falls_back_to_the_environment(monkeypatch, registry_dir):
    monkeypatch.setenv("AUDITOR_STT_REGISTRY_DIR", str(registry_dir))
    app = create_app(model_host=_Host("stub", preloaded=True), queue=InferenceQueue(), api_key=KEY)
    with TestClient(app) as client:
        assert client.get("/model", headers=AUTH).json()["registry"][0]["name"] == "whisper-fi"


def test_health_and_status_bodies_are_unchanged_by_the_registry(registry_dir):
    app = _app(_Host("stub", preloaded=True), _Factory(), registry_dir)
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok", "model_id": "stub", "device": "cpu", "compute_type": "int8", "loaded": True,
        }


# --- POST /model: who may switch, and to what -----------------------------------------------


def test_switching_is_forbidden_when_no_api_key_is_configured(registry_dir):
    old, factory = _Host("old", preloaded=True), _Factory(_Host("new"))
    app = _app(old, factory, registry_dir, api_key=None, allowed=["large-v3-turbo"])
    with TestClient(app) as client:
        for spec in ("registry:whisper-fi", "large-v3-turbo", "/etc"):
            resp = _switch(client, spec, headers={})
            assert resp.status_code == 403
            assert resp.json() == {"detail": "Model switching requires AUDITOR_STT_API_KEY"}
        # The check comes before the body is looked at, and a bearer token does not help without a configured key.
        assert client.post("/model", json={}, headers=AUTH).status_code == 403
        assert SWITCH_NEEDS_KEY in client.post("/model", headers=AUTH).text
        assert app.state.model_host is old
    assert factory.requested == []


def test_switching_needs_the_correct_key(registry_dir):
    factory = _Factory(_Host("new"))
    app = _app(_Host("old", preloaded=True), factory, registry_dir)
    with TestClient(app) as client:
        assert _switch(client, "registry:whisper-fi", headers={}).status_code == 401
        resp = _switch(client, "registry:whisper-fi", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"
        assert client.get("/health").json()["model_id"] == "old"
    assert factory.requested == []


@pytest.mark.parametrize(
    "spec",
    [
        "/etc",
        "/models/my-ct2",
        "C:\\models\\my-ct2",
        "../../models",
        "./models/registry/whisper-fi/v1/ct2",
        "large-v3-turbo",  # a real alias, but not on the allow-list
        "Systran/faster-whisper-small",
        "REGISTRY:whisper-fi",
        " ",
    ],
)
def test_raw_paths_and_unlisted_aliases_are_rejected(registry_dir, spec):
    old, factory = _Host("old", preloaded=True), _Factory(_Host("new"))
    app = _app(old, factory, registry_dir, allowed=["only-this-alias"])
    with TestClient(app) as client:
        resp = _switch(client, spec)
        assert resp.status_code == 422
        assert "AUDITOR_STT_ALLOWED_MODELS" in resp.json()["detail"]
        assert app.state.model_host is old
    assert factory.requested == []


def test_an_allow_listed_alias_can_be_switched_to(registry_dir):
    new = _Host("large-v3-turbo")
    factory = _Factory(new)
    app = _app(_Host("old", preloaded=True), factory, registry_dir, allowed=["small", "large-v3-turbo"])
    with TestClient(app) as client:
        resp = _switch(client, "large-v3-turbo")

        assert resp.status_code == 200
        assert resp.json() == {
            "model": {
                "model_id": "large-v3-turbo", "device": "cpu", "compute_type": "int8", "loaded": True,
                "source": "large-v3-turbo",
            }
        }
        assert factory.requested == ["large-v3-turbo"]  # aliases reach the factory unchanged
        assert client.get("/model", headers=AUTH).json()["model"]["source"] == "large-v3-turbo"


def test_the_allow_list_is_read_from_the_environment(monkeypatch, registry_dir):
    monkeypatch.setenv("AUDITOR_STT_ALLOWED_MODELS", " small , large-v3-turbo ,, ")
    factory = _Factory(_Host("small"), _Host("large-v3-turbo"))
    app = create_app(
        model_host=_Host("old", preloaded=True), queue=InferenceQueue(), api_key=KEY,
        host_factory=factory, registry_dir=registry_dir,
    )
    with TestClient(app) as client:
        assert _switch(client, "small").status_code == 200
        assert _switch(client, "large-v3-turbo").status_code == 200
        assert _switch(client, "medium").status_code == 422
    assert factory.requested == ["small", "large-v3-turbo"]


def test_an_empty_allow_list_allows_only_the_registry(monkeypatch, registry_dir):
    monkeypatch.setenv("AUDITOR_STT_ALLOWED_MODELS", "")
    app = create_app(
        model_host=_Host("old", preloaded=True), queue=InferenceQueue(), api_key=KEY,
        host_factory=_Factory(_Host("new")), registry_dir=registry_dir,
    )
    with TestClient(app) as client:
        assert _switch(client, "large-v3-turbo").status_code == 422
        assert _switch(client, "registry:whisper-fi").status_code == 200


def test_a_registry_spec_switches_to_the_current_version(registry_dir):
    new = _Host("whatever-the-factory-named-it")
    factory = _Factory(new)
    app = _app(_Host("old", preloaded=True), factory, registry_dir)
    with TestClient(app) as client:
        resp = _switch(client, "registry:whisper-fi")

        assert resp.status_code == 200
        assert factory.requested == [str(registry_dir / "whisper-fi" / "v2" / "ct2")]
        # Reported by label, not by directory: no server path in /health, /model or job results.
        assert resp.json()["model"]["model_id"] == "registry:whisper-fi@v2"
        assert resp.json()["model"]["source"] == "registry:whisper-fi"
        assert client.get("/health").json()["model_id"] == "registry:whisper-fi@v2"
        assert str(registry_dir) not in client.get("/health").text


def test_a_registry_spec_can_name_a_version(registry_dir):
    factory = _Factory(_Host("x"))
    app = _app(_Host("old", preloaded=True), factory, registry_dir)
    with TestClient(app) as client:
        resp = _switch(client, "registry:whisper-fi@v1")
        assert resp.status_code == 200
        assert resp.json()["model"]["model_id"] == "registry:whisper-fi@v1"
        assert resp.json()["model"]["source"] == "registry:whisper-fi@v1"
    assert factory.requested == [str(registry_dir / "whisper-fi" / "v1" / "ct2")]


@pytest.mark.parametrize(
    ("spec", "detail"),
    [
        ("registry:nobody", "no promoted version"),
        ("registry:nobody@v1", "nobody@v1 is not registered"),
        ("registry:whisper-fi@v9", "whisper-fi@v9 is not registered"),
        ("registry:../whisper-fi", "Invalid model name"),
        ("registry:", "Invalid model name"),
        ("registry:whisper-fi@../v1", "Invalid version"),
        ("registry:whisper-fi@", "empty version"),
    ],
)
def test_an_unknown_or_malformed_registry_spec_is_a_422(registry_dir, spec, detail):
    old, factory = _Host("old", preloaded=True), _Factory(_Host("new"))
    app = _app(old, factory, registry_dir)
    with TestClient(app) as client:
        resp = _switch(client, spec)
        assert resp.status_code == 422
        assert detail in resp.json()["detail"]
        assert str(registry_dir) not in resp.text
        assert app.state.model_host is old
    assert factory.requested == []


@pytest.mark.parametrize("body", [{}, {"model": ""}, {"model": 5}, {"other": "registry:whisper-fi"}])
def test_a_malformed_body_is_a_422(registry_dir, body):
    factory = _Factory(_Host("new"))
    app = _app(_Host("old", preloaded=True), factory, registry_dir)
    with TestClient(app) as client:
        assert client.post("/model", json=body, headers=AUTH).status_code == 422
    assert factory.requested == []


# --- POST /model: the swap ---------------------------------------------------------------------


def test_a_successful_switch_serves_later_requests_from_the_new_host_and_frees_the_old_one(registry_dir):
    old, new = _Host("old", preloaded=True), _Host("new")
    old_ref = weakref.ref(old)
    app = _app(old, _Factory(new), registry_dir, allowed=["new"])
    del old
    with TestClient(app) as client:
        assert _transcribe(client).json()["text"] == "from old"

        assert _switch(client, "new").status_code == 200

        assert _transcribe(client).json()["text"] == "from new"
        assert client.get("/health").json()["model_id"] == "new"
        assert client.get("/status", headers=AUTH).json()["model"]["model_id"] == "new"
        gc.collect()
        assert old_ref() is None  # nothing holds the old host any more


def test_a_failing_load_keeps_the_old_host_serving(registry_dir, caplog):
    caplog.set_level(logging.ERROR, logger="auditor_stt.serve.app")
    old = _Host("old", preloaded=True)
    broken = _Host("broken", load_error=ModelLoadError("Could not load /srv/secret/place on any device"))
    app = _app(old, _Factory(broken), registry_dir, allowed=["broken"])
    with TestClient(app) as client:
        resp = _switch(client, "broken")

        assert resp.status_code == 500
        assert resp.json() == {"detail": "Could not load model 'broken'; the current model is still serving"}
        assert "/srv/secret" not in resp.text  # the exception text stays in the log
        assert "/srv/secret" in caplog.text
        assert app.state.model_host is old
        assert client.get("/health").json()["model_id"] == "old"
        assert _transcribe(client).json()["text"] == "from old"
        assert client.get("/model", headers=AUTH).json()["model"]["source"] == "old"


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("cuda out of memory"),
        FileNotFoundError("model.bin"),
        ValueError("bad config"),
    ],
)
def test_any_load_error_is_a_500_and_keeps_the_old_host(registry_dir, failure):
    old = _Host("old", preloaded=True)
    app = _app(old, _Factory(_Host("broken", load_error=failure)), registry_dir, allowed=["broken"])
    with TestClient(app) as client:
        assert _switch(client, "broken").status_code == 500
        assert app.state.model_host is old


def test_a_factory_that_raises_is_a_500_and_keeps_the_old_host(registry_dir):
    def factory(_model_id):
        raise RuntimeError("cannot even build a host")

    old = _Host("old", preloaded=True)
    app = _app(old, factory, registry_dir, allowed=["broken"])
    with TestClient(app) as client:
        resp = _switch(client, "broken")
        assert resp.status_code == 500
        assert "cannot even build" not in resp.text
        assert app.state.model_host is old


def test_a_host_that_is_not_loaded_after_load_is_not_swapped_in(registry_dir):
    class _Lazy(_Host):
        def load(self):
            pass  # returns without raising but never becomes loaded

    old = _Host("old", preloaded=True)
    app = _app(old, _Factory(_Lazy("lazy")), registry_dir, allowed=["lazy"])
    with TestClient(app) as client:
        assert _switch(client, "lazy").status_code == 500
        assert app.state.model_host is old


def test_a_failed_switch_does_not_block_the_next_one(registry_dir):
    broken = _Host("broken", load_error=ModelLoadError("nope"))
    good = _Host("good")
    app = _app(_Host("old", preloaded=True), _Factory(broken, good), registry_dir, allowed=["broken", "good"])
    with TestClient(app) as client:
        assert _switch(client, "broken").status_code == 500
        assert _switch(client, "good").status_code == 200
        assert client.get("/health").json()["model_id"] == "good"


def test_a_request_in_flight_during_the_switch_finishes_on_the_old_host(registry_dir):
    old, new = _Host("old", preloaded=True), _Host("new")
    old.release = threading.Event()
    app = _app(old, _Factory(new), registry_dir, allowed=["new"])
    with TestClient(app) as client:
        thread, result = _in_thread(lambda: _transcribe(client))
        try:
            _until(old.entered.is_set)  # the request is inside the old model now

            switched = _switch(client, "new")

            assert switched.status_code == 200
            assert client.get("/health").json()["model_id"] == "new"  # new requests already see the new model
            assert "value" not in result  # ... while the old request has not finished
        finally:
            old.release.set()
        thread.join(10)

        assert result["value"].status_code == 200
        assert result["value"].json()["text"] == "from old"  # it finished on the host it started with
        assert _transcribe(client).json()["text"] == "from new"


def test_a_second_switch_while_one_is_loading_is_a_409(registry_dir):
    slow, third = _Host("slow"), _Host("third")
    slow.load_release = threading.Event()
    factory = _Factory(slow, third)
    old = _Host("old", preloaded=True)
    app = _app(old, factory, registry_dir, allowed=["slow", "third"])
    with TestClient(app) as client:
        thread, result = _in_thread(lambda: _switch(client, "slow"))
        try:
            _until(slow.load_started.is_set)

            resp = _switch(client, "third")

            assert resp.status_code == 409
            assert resp.json() == {"detail": "A model switch is already in progress"}
            assert factory.requested == ["slow"]  # the refused request loaded nothing
            assert app.state.model_host is old  # and the running one is not swapped in early
            assert _transcribe(client).json()["text"] == "from old"  # the old model keeps serving during the load
        finally:
            slow.load_release.set()
        thread.join(10)

        assert result["value"].status_code == 200
        assert client.get("/health").json()["model_id"] == "slow"
        assert _switch(client, "third").status_code == 200  # the lock was released
        assert client.get("/health").json()["model_id"] == "third"


def test_the_switch_is_logged_and_a_device_change_is_called_out(registry_dir, caplog):
    caplog.set_level(logging.INFO, logger="auditor_stt.serve.app")
    old = _Host("old", preloaded=True)
    new = _Host("new", device="cuda", compute_type="float16")
    app = _app(old, _Factory(new), registry_dir, allowed=["new"])
    with TestClient(app) as client:
        assert _switch(client, "new").status_code == 200
    assert "Switched model from old to new (cuda/float16)" in caplog.text
    assert "runs on cuda/float16, not cpu/int8" in caplog.text


def test_no_device_warning_when_the_device_stays_the_same(registry_dir, caplog):
    caplog.set_level(logging.INFO, logger="auditor_stt.serve.app")
    app = _app(_Host("old", preloaded=True), _Factory(_Host("new")), registry_dir, allowed=["new"])
    with TestClient(app) as client:
        assert _switch(client, "new").status_code == 200
    assert "Switched model" in caplog.text
    assert "runs on" not in caplog.text


# --- startup with AUDITOR_STT_MODEL ---------------------------------------------------------


def test_startup_uses_the_default_model_alias_when_none_is_configured(registry_dir):
    factory = _Factory(_Host("x"))
    app = create_app(queue=InferenceQueue(), host_factory=factory, registry_dir=registry_dir)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model").json()["model"]["source"] == DEFAULT_MODEL
    assert factory.requested == [DEFAULT_MODEL]


def test_startup_can_resolve_a_registry_spec(monkeypatch, registry_dir):
    monkeypatch.setenv("AUDITOR_STT_MODEL", "registry:whisper-fi")
    factory = _Factory(_Host("named-by-path"))
    app = create_app(queue=InferenceQueue(), host_factory=factory, registry_dir=registry_dir)
    with TestClient(app) as client:
        health = client.get("/health")

        assert health.status_code == 200
        assert health.json()["model_id"] == "registry:whisper-fi@v2"
        assert client.get("/model").json()["model"]["source"] == "registry:whisper-fi"
        assert _transcribe(client, headers={}).json()["text"] == "from named-by-path"  # no key configured
    assert factory.requested == [str(registry_dir / "whisper-fi" / "v2" / "ct2")]


def test_startup_can_resolve_a_specific_registry_version(monkeypatch, registry_dir):
    monkeypatch.setenv("AUDITOR_STT_MODEL", "registry:whisper-fi@v1")
    factory = _Factory(_Host("x"))
    app = create_app(queue=InferenceQueue(), host_factory=factory, registry_dir=registry_dir)
    with TestClient(app) as client:
        assert client.get("/health").json()["model_id"] == "registry:whisper-fi@v1"
    assert factory.requested == [str(registry_dir / "whisper-fi" / "v1" / "ct2")]


def test_startup_takes_an_arbitrary_alias_or_path_from_the_environment(monkeypatch, registry_dir):
    # The allow-list only guards POST /model; the operator's own AUDITOR_STT_MODEL may be anything.
    monkeypatch.setenv("AUDITOR_STT_MODEL", "/srv/models/my-ct2")
    factory = _Factory(_Host("x"))
    create_app(queue=InferenceQueue(), host_factory=factory, registry_dir=registry_dir)
    assert factory.requested == ["/srv/models/my-ct2"]


def test_a_registry_model_that_fails_to_load_at_startup_leaves_health_at_503(monkeypatch, registry_dir):
    monkeypatch.setenv("AUDITOR_STT_MODEL", "registry:whisper-fi")
    broken = _Host("x", load_error=ModelLoadError("boom"))
    app = create_app(queue=InferenceQueue(), host_factory=_Factory(broken), registry_dir=registry_dir)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["model_id"] == "registry:whisper-fi@v2"  # not the directory
        assert health.json()["loaded"] is False


@pytest.mark.parametrize("spec", ["registry:nobody", "registry:whisper-fi@v9", "registry:../x", "registry:"])
def test_a_broken_registry_spec_does_not_stop_the_app_and_health_reports_loading(
    monkeypatch, registry_dir, caplog, spec
):
    caplog.set_level(logging.ERROR, logger="auditor_stt.serve.app")
    monkeypatch.setenv("AUDITOR_STT_MODEL", spec)
    factory = _Factory()  # a call would raise IndexError: nothing may be built for a spec that cannot be resolved

    app = create_app(queue=InferenceQueue(), api_key=KEY, host_factory=factory, registry_dir=registry_dir)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "loading" and health.json()["loaded"] is False
        assert "Cannot use AUDITOR_STT_MODEL" in caplog.text
        assert _transcribe(client).status_code == 503
        model = client.get("/model", headers=AUTH).json()["model"]
        assert model["loaded"] is False and model["source"] == spec
    assert factory.requested == []


def test_a_service_started_with_a_broken_model_can_be_fixed_by_a_switch(monkeypatch, registry_dir):
    monkeypatch.setenv("AUDITOR_STT_MODEL", "registry:nobody")
    app = create_app(
        queue=InferenceQueue(), api_key=KEY, host_factory=_Factory(_Host("fixed")), registry_dir=registry_dir
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503

        assert _switch(client, "registry:whisper-fi").status_code == 200

        assert client.get("/health").status_code == 200
        assert _transcribe(client).json()["text"] == "from fixed"


# --- job consistency ---------------------------------------------------------------------------


def _write_wav(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16_wav(path, np.zeros(int(seconds * RATE), dtype=np.float32), RATE)
    return path


def _no_gap(_samples, _rate):
    return None


def _copy_normalize(source, wav_path, *, cancel=None):
    shutil.copyfile(source, wav_path)


def test_transcribe_chunk_records_the_model_that_produced_it(tmp_path):
    from auditor_stt.serve.jobs.chunking import Chunk

    path = _write_wav(tmp_path / "a.wav", 10)
    with open_pcm16_mono(path) as wav:
        result = transcribe_chunk(
            _Host("registry:whisper-fi@v2", preloaded=True), wav, Chunk(0, 0, 5 * RATE, True),
            language="fi", word_timestamps=True, prompt=None, vad=True, previous_text=None, carry_chars=0,
        )
    assert result["model"] == "registry:whisper-fi@v2"
    assert set(result) == {"text", "language", "segments", "model"}


def _chunk(text, model):
    result = {"text": text, "language": "fi", "segments": []}
    if model is not None:
        result["model"] = model
    return result


def _manifest(chunks):
    plan = [{"index": i, "start": i * 60.0, "end": (i + 1) * 60.0, "start_sample": 0, "end_sample": 0, "snapped": True}
            for i in range(chunks)]
    return {"params": {"language": "fi"}, "chunks": plan}


def test_assemble_lists_the_distinct_models_in_chunk_order():
    results = {0: _chunk("a", "old"), 1: _chunk("b", "old"), 2: _chunk("c", "new"), 3: _chunk("d", "old")}
    assembled = assemble(_manifest(4), {3: results[3], 1: results[1], 2: results[2], 0: results[0]})
    assert assembled["models"] == ["old", "new"]  # first appearance in chunk order, each once
    assert assembled["text"] == "a b c d"  # every existing key keeps its meaning


def test_assemble_partial_lists_only_the_models_of_finished_chunks():
    assembled = assemble(_manifest(3), {0: _chunk("a", "old"), 2: _chunk("c", "new"), 1: None}, partial=True)
    assert assembled["models"] == ["old", "new"]
    assert assemble(_manifest(3), {0: _chunk("a", "old")}, partial=True)["models"] == ["old"]


def test_assemble_leaves_models_out_when_no_chunk_names_one():
    # Chunk files written before models were recorded, or by a host without a model_id.
    assembled = assemble(_manifest(2), {0: _chunk("a", None), 1: _chunk("b", None)})
    assert "models" not in assembled
    mixed = assemble(_manifest(2), {0: _chunk("a", None), 1: _chunk("b", "new")})
    assert mixed["models"] == ["new"]


def test_a_job_that_spans_a_model_switch_says_so(tmp_path):
    old, new = _Host("old", preloaded=True), _Host("new")
    reached, release = threading.Event(), threading.Event()

    def hold_chunk_two(offset):
        if offset == 10.0:  # chunk 2 of six is running on the old model when the switch happens
            reached.set()
            release.wait(10)

    old.hook = hold_chunk_two
    queue = InferenceQueue()
    store = JobStore(tmp_path / "data" / "jobs")
    app_ref = {}
    runner = JobRunner(
        store,
        lambda: app_ref["app"].state.model_host,
        queue,
        RunnerConfig(retry_backoff=(0.0, 0.0), model_poll_seconds=0.01),
        normalize=_copy_normalize,
        find_gap=_no_gap,
    )
    media = tmp_path / "media"
    _write_wav(media / "talk.wav", 30)
    app = create_app(
        model_host=old, queue=queue, api_key=KEY, data_dir=tmp_path / "data", media_root=media,
        job_store=store, runner=runner, host_factory=_Factory(new), registry_dir=tmp_path / "registry",
        allowed_models=["new"],
    )
    app_ref["app"] = app

    with TestClient(app) as client:
        submitted = client.post(
            "/v1/jobs", data={"source_path": "talk.wav", "chunk_seconds": "5"}, headers=AUTH
        )
        assert submitted.status_code == 202, submitted.text
        job_id = submitted.json()["id"]
        try:
            _until(reached.is_set)
            assert _switch(client, "new").status_code == 200
        finally:
            release.set()
        _until(lambda: client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()["status"] == "completed")

        chunk_models = [store.read_chunk(job_id, index)["model"] for index in range(6)]
        # Chunk 2 was already running on the old model and finishes there; later chunks use the new one.
        assert chunk_models == ["old", "old", "old", "new", "new", "new"]

        result = client.get(f"/v1/jobs/{job_id}/result", headers=AUTH).json()
        assert result["models"] == ["old", "new"]
        assert result["complete"] is True and result["chunks_done"] == 6
        assert [segment["start"] for segment in result["segments"]] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
