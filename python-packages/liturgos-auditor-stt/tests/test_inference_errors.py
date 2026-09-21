from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.model import AudioDecodeError, ModelHost, ModelLoadError
from auditor_stt.serve.queue import InferenceQueue


class _StubHost:
    model_id = "stub-model"
    device = "cpu"
    compute_type = "int8"
    loaded = True

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def load(self):
        pass

    def transcribe(self, audio_path, language=None):
        if self._error:
            raise self._error
        return self._result


def _post(host, **kwargs):
    app = create_app(model_host=host, queue=InferenceQueue(max_queue=8))
    with TestClient(app) as client:
        return client.post("/inference", files={"file": ("segment.mp4", b"bytes", "audio/mp4")}, **kwargs)


def test_undecodable_audio_returns_422_with_json_detail():
    resp = _post(_StubHost(error=AudioDecodeError("Could not decode audio")))
    assert resp.status_code == 422
    assert resp.json() == {"detail": "Could not decode audio"}


def test_silence_returns_200_with_empty_text():
    # lcyt drops blank text silently; an error status would surface as an error event there.
    silence = {"text": "", "language": "fi", "segments": []}
    resp = _post(_StubHost(result=silence))
    assert resp.status_code == 200
    assert resp.json()["text"] == ""


def test_startup_survives_model_load_failure_and_health_reports_loading():
    # Regression: app.py logged the failure through an undefined `logger`,
    # so a model that failed to load crashed startup with NameError.
    class _FailingHost(_StubHost):
        loaded = False

        def load(self):
            raise ModelLoadError("no such model")

    app = create_app(model_host=_FailingHost(), queue=InferenceQueue(max_queue=8))
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "loading"


def test_model_maps_real_decode_failure_to_audio_decode_error(monkeypatch, tmp_path):
    import faster_whisper
    from faster_whisper.audio import decode_audio

    class _DecodingModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            if not isinstance(audio, str):
                # load() warms the model up with a silent sample array, which needs no decoding.
                return iter([]), SimpleNamespace(language="en")
            decode_audio(audio)  # what faster-whisper does first for a path; raises av.error.InvalidDataError
            raise AssertionError("garbage must not decode")

    monkeypatch.setattr(faster_whisper, "WhisperModel", _DecodingModel)
    host = ModelHost("test-model", device="cpu")
    host.load()

    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not audio at all " * 200)

    with pytest.raises(AudioDecodeError) as excinfo:
        host.transcribe(str(garbage), language="fi")

    # The raw PyAV message embeds the server's temp path; the client-facing one must not.
    assert str(tmp_path) not in str(excinfo.value)
