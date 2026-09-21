from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.model import ModelHost
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
        "words": [{"start": 0.0, "end": 0.4, "text": "moi", "probability": 0.99}],
    }],
}


class _RecordingHost:
    model_id = "stub-model"
    device = "cpu"
    compute_type = "int8"
    loaded = True

    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def transcribe(self, audio_path, language=None, **options):
        self.calls.append({"language": language, **options})
        return RESULT


class _LegacyHost:
    """Signature from before the options existed: any extra kwarg would raise TypeError."""

    model_id = "stub-model"
    device = "cpu"
    compute_type = "int8"
    loaded = True

    def load(self):
        pass

    def transcribe(self, audio_path, language=None):
        return RESULT


def _post(host, **fields):
    app = create_app(model_host=host, queue=InferenceQueue(max_queue=8))
    with TestClient(app) as client:
        return client.post(
            "/inference",
            files={"file": ("clip.wav", b"not-real-wav-bytes", "audio/wav")},
            data=fields,
        )


# --- /inference option forwarding -------------------------------------------------------------


def test_no_options_are_forwarded_by_default():
    host = _RecordingHost()
    resp = _post(host)
    assert resp.status_code == 200
    assert host.calls == [{"language": "fi"}]


def test_legacy_host_signature_keeps_working_without_options():
    assert _post(_LegacyHost()).json() == RESULT
    assert _post(_LegacyHost(), model="whisper-1", response_format="text").text == "moi maailma"


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"prompt": "Herra armahda"}, {"prompt": "Herra armahda"}),
        ({"vad": "true"}, {"vad": True}),
        ({"vad": "false"}, {"vad": False}),
        ({"temperature": "0.2"}, {"temperature": 0.2}),
        (
            {"prompt": "Amen", "vad": "1", "temperature": "0"},
            {"prompt": "Amen", "vad": True, "temperature": 0.0},
        ),
        ({"prompt": "", "temperature": "", "vad": ""}, {}),  # blank form fields count as not supplied
    ],
)
def test_only_supplied_options_are_forwarded(fields, expected):
    host = _RecordingHost()
    assert _post(host, **fields).status_code == 200
    assert host.calls == [{"language": "fi", **expected}]


def test_language_and_options_are_forwarded_together():
    host = _RecordingHost()
    _post(host, language="en", prompt="Amen")
    assert host.calls == [{"language": "en", "prompt": "Amen"}]


def test_model_field_is_accepted_and_ignored():
    host = _RecordingHost()
    assert _post(host, model="large-v3").status_code == 200
    assert host.calls == [{"language": "fi"}]


def test_malformed_option_is_a_422():
    host = _RecordingHost()
    assert _post(host, temperature="hot").status_code == 422
    assert host.calls == []


# --- /inference response_format ---------------------------------------------------------------


def test_json_is_the_default_response_format():
    resp = _post(_RecordingHost())
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == RESULT


def test_verbose_json_is_the_same_as_json():
    host = _RecordingHost()
    assert _post(host, response_format="verbose_json").json() == _post(host).json()


def test_text_format_returns_plain_text_body():
    resp = _post(_RecordingHost(), response_format="text")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "moi maailma"


@pytest.mark.parametrize("response_format", ["srt", "vtt", "nonsense"])
def test_unsupported_response_format_is_a_400_and_never_transcribes(response_format):
    host = _RecordingHost()
    resp = _post(host, response_format=response_format)
    assert resp.status_code == 400
    assert "response_format" in resp.json()["detail"]
    assert host.calls == []


# --- ModelHost: option mapping, result shape, transcribe_array --------------------------------


class _RecordingWhisperModel:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, "kwargs": kwargs})
        word = SimpleNamespace(start=1.1, end=1.5, word=" hei", probability=0.9)
        segment = SimpleNamespace(
            start=1.0, end=2.0, text=" hei ", words=[word], avg_logprob=-0.3, no_speech_prob=0.02
        )
        return iter([segment]), SimpleNamespace(language="fi")


@pytest.fixture
def host(monkeypatch):
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", _RecordingWhisperModel)
    model_host = ModelHost("test-model", device="cpu")
    model_host.load()
    # load() warms the model up with a silent transcribe; only calls made by the test count.
    model_host.model.calls.clear()
    return model_host


def test_model_passes_no_extra_kwargs_by_default(host):
    host.transcribe("clip.wav", language="fi")
    assert host.model.calls[0]["kwargs"] == {"language": "fi", "word_timestamps": True}


def test_model_maps_options_onto_faster_whisper_kwargs(host):
    host.transcribe(
        "clip.wav",
        language="fi",
        prompt="Herra armahda",
        vad=True,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    assert host.model.calls[0]["kwargs"] == {
        "language": "fi",
        "word_timestamps": False,
        "initial_prompt": "Herra armahda",
        "vad_filter": True,
        "temperature": 0.0,
        "condition_on_previous_text": False,
    }


def test_model_leaves_falsy_defaults_to_faster_whisper(host):
    host.transcribe("clip.wav", language="fi", prompt="", vad=False, temperature=None)
    assert host.model.calls[0]["kwargs"] == {"language": "fi", "word_timestamps": True}


def test_segments_carry_confidence_fields(host):
    segment = host.transcribe("clip.wav", language="fi")["segments"][0]
    assert segment["avg_logprob"] == -0.3
    assert segment["no_speech_prob"] == 0.02


def test_segments_without_confidence_fields_report_none(monkeypatch):
    import faster_whisper

    class _BareModel(_RecordingWhisperModel):
        def transcribe(self, audio, **kwargs):
            segment = SimpleNamespace(start=0.0, end=1.0, text="hei", words=None)
            return iter([segment]), SimpleNamespace(language="fi")

    monkeypatch.setattr(faster_whisper, "WhisperModel", _BareModel)
    model_host = ModelHost("test-model", device="cpu")
    model_host.load()

    segment = model_host.transcribe("clip.wav", language="fi")["segments"][0]
    assert segment["avg_logprob"] is None
    assert segment["no_speech_prob"] is None
    assert segment["words"] == []


def test_transcribe_array_hands_the_samples_straight_to_faster_whisper(host):
    samples = np.zeros(16000, dtype=np.float32)
    host.transcribe_array(samples, language="fi", prompt="Amen", vad=True)
    call = host.model.calls[0]
    assert call["audio"] is samples
    assert call["kwargs"] == {
        "language": "fi",
        "word_timestamps": True,
        "initial_prompt": "Amen",
        "vad_filter": True,
    }


def test_transcribe_array_shapes_the_result_like_transcribe(host):
    samples = np.zeros(16000, dtype=np.float32)
    assert host.transcribe_array(samples, language="fi") == host.transcribe("clip.wav", language="fi")


def test_transcribe_array_offsets_segment_and_word_timestamps(host):
    samples = np.zeros(16000, dtype=np.float32)
    result = host.transcribe_array(samples, language="fi", time_offset=60.0)
    segment = result["segments"][0]
    assert (segment["start"], segment["end"]) == (61.0, 62.0)
    assert (segment["words"][0]["start"], segment["words"][0]["end"]) == (61.1, 61.5)
    assert segment["text"] == "hei"
    assert result["text"] == "hei"


def test_transcribe_array_requires_a_loaded_model():
    with pytest.raises(RuntimeError):
        ModelHost("test-model", device="cpu").transcribe_array(np.zeros(16000, dtype=np.float32))
