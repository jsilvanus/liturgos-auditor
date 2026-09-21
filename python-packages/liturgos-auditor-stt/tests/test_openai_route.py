import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.model import AudioDecodeError
from auditor_stt.serve.queue import InferenceQueue

RESULT = {
    "text": "moi maailma",
    "language": "fi",
    "segments": [{
        "start": 0.0,
        "end": 1.5,
        "text": "moi maailma",
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
        "words": [
            {"start": 0.0, "end": 0.4, "text": "moi", "probability": 0.99},
            {"start": 0.5, "end": 1.5, "text": " maailma", "probability": 0.98},
        ],
    }],
}


class _StubHost:
    model_id = "stub-model"
    device = "cpu"
    compute_type = "int8"
    loaded = True

    def __init__(self, result=RESULT, error=None):
        self.calls = []
        self._result = result
        self._error = error

    def load(self):
        pass

    def transcribe(self, audio_path, language=None, **options):
        self.calls.append({"language": language, **options})
        if self._error:
            raise self._error
        return self._result


def _post(host=None, queue=None, **fields):
    host = host or _StubHost()
    app = create_app(model_host=host, queue=queue or InferenceQueue(max_queue=8))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", b"not-real-wav-bytes", "audio/wav")},
            data=fields,
        )
    return resp, host


def test_json_is_the_default_and_carries_only_text():
    resp, _host = _post(model="whisper-1")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"text": "moi maailma"}


def test_language_defaults_to_the_service_language_and_can_be_overridden():
    _resp, host = _post()
    assert host.calls == [{"language": "fi"}]

    _resp, host = _post(language="en")
    assert host.calls == [{"language": "en"}]


def test_prompt_and_temperature_are_forwarded_only_when_supplied():
    _resp, host = _post(prompt="Herra armahda", temperature="0.2")
    assert host.calls == [{"language": "fi", "prompt": "Herra armahda", "temperature": 0.2}]

    _resp, host = _post(model="whisper-1")
    assert host.calls == [{"language": "fi"}]


def test_text_format_returns_plain_text_body():
    resp, _host = _post(response_format="text")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "moi maailma"


def test_verbose_json_returns_the_full_result():
    resp, _host = _post(response_format="verbose_json")
    assert resp.status_code == 200
    assert resp.json() == RESULT


@pytest.mark.parametrize("response_format", ["srt", "vtt", "nonsense"])
def test_unsupported_formats_are_rejected_before_transcribing(response_format):
    resp, host = _post(response_format=response_format)
    assert resp.status_code == 400
    assert "response_format" in resp.json()["detail"]
    assert host.calls == []


def test_undecodable_audio_returns_422():
    resp, _host = _post(_StubHost(error=AudioDecodeError("Could not decode audio")))
    assert resp.status_code == 422
    assert resp.json() == {"detail": "Could not decode audio"}


def test_silence_returns_200_with_empty_text():
    silence = {"text": "", "language": "fi", "segments": []}
    resp, _host = _post(_StubHost(result=silence))
    assert resp.status_code == 200
    assert resp.json() == {"text": ""}


def test_returns_503_when_model_not_loaded():
    class _NotLoaded(_StubHost):
        loaded = False

    resp, host = _post(_NotLoaded())
    assert resp.status_code == 503
    assert host.calls == []


def test_returns_503_when_the_live_queue_is_full():
    resp, host = _post(queue=InferenceQueue(max_queue=0))
    assert resp.status_code == 503
    assert host.calls == []


def test_missing_file_is_a_422():
    app = create_app(model_host=_StubHost(), queue=InferenceQueue(max_queue=8))
    with TestClient(app) as client:
        resp = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert resp.status_code == 422
