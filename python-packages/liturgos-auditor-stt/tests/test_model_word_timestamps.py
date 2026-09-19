from types import SimpleNamespace

from auditor_stt.serve.model import ModelHost


class _FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def transcribe(self, audio_path, **kwargs):
        assert kwargs["word_timestamps"] is True
        word = SimpleNamespace(start=0.2, end=0.7, word=" maailma", probability=0.97)
        segment = SimpleNamespace(start=0.0, end=1.0, text="maailma", words=[word])
        info = SimpleNamespace(language="fi")
        return iter([segment]), info


def test_transcribe_requests_and_returns_word_timestamps(monkeypatch):
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)

    host = ModelHost("test-model", device="cpu")
    host.load()

    result = host.transcribe("/tmp/audio.wav", language="fi")

    assert result["language"] == "fi"
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 1.0
    assert result["segments"][0]["words"] == [{
        "start": 0.2,
        "end": 0.7,
        "text": " maailma",
        "probability": 0.97,
    }]
