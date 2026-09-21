"""Test model warm-up fallback behavior on device failure."""

from types import SimpleNamespace

import pytest

from auditor_stt.serve import model as model_module
from auditor_stt.serve.model import ModelHost, ModelLoadError


class _FakeWhisperModel:
    """A fake model that succeeds on transcribe with word_timestamps=True."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def transcribe(self, audio_path, **kwargs):
        assert kwargs["word_timestamps"] is True
        word = SimpleNamespace(start=0.0, end=1.0, word=" ", probability=1.0)
        segment = SimpleNamespace(start=0.0, end=1.0, text="", words=[word])
        info = SimpleNamespace(language="en")
        return iter([segment]), info


class _FakeWhisperModelFailsOnWarmup:
    """A fake model that fails when transcribe is called (during warm-up)."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def transcribe(self, audio_path, **kwargs):
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


def test_warm_up_not_failing_leaves_device_unchanged(monkeypatch):
    """When warm-up succeeds, device and compute_type are set correctly."""
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)

    host = ModelHost("test-model", device="cpu")
    host.load()

    assert host.loaded is True
    assert host.device == "cpu"
    assert host.compute_type == "int8"


def test_warm_up_failure_on_cuda_auto_falls_back_to_cpu(monkeypatch, caplog):
    """When CUDA warm-up fails with device='auto', fall back to CPU and log warning."""
    import logging

    import faster_whisper

    call_count = {}

    class _FailCudaThenSucceedCpu:
        def __init__(self, *args, device=None, compute_type=None, **kwargs):
            self.device = device
            self.compute_type = compute_type
            self.kwargs = kwargs

        def transcribe(self, audio_path, **kwargs):
            assert kwargs["word_timestamps"] is True
            # Fail on CUDA, succeed on CPU.
            if self.device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            word = SimpleNamespace(start=0.0, end=1.0, word=" ", probability=1.0)
            segment = SimpleNamespace(start=0.0, end=1.0, text="", words=[word])
            info = SimpleNamespace(language="en")
            return iter([segment]), info

    monkeypatch.setattr(faster_whisper, "WhisperModel", _FailCudaThenSucceedCpu)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        host = ModelHost("test-model", device="auto")
        host.load()

    assert host.loaded is True
    assert host.device == "cpu"
    assert host.compute_type == "int8"
    # Check that the warning was logged with the exception class name and message.
    assert any("Failed to load model" in record.message and "RuntimeError" in record.message for record in caplog.records)


def test_explicit_cuda_device_raises_on_warm_up_failure(monkeypatch):
    """When device='cuda' is explicit and warm-up fails, raise ModelLoadError immediately."""
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModelFailsOnWarmup)

    host = ModelHost("test-model", device="cuda")
    with pytest.raises(ModelLoadError, match="Could not load model"):
        host.load()

    assert host.loaded is False


def test_explicit_cpu_device_succeeds(monkeypatch):
    """When device='cpu' is explicit and warm-up succeeds, load succeeds."""
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)

    host = ModelHost("test-model", device="cpu")
    host.load()

    assert host.loaded is True
    assert host.device == "cpu"


def test_warm_up_call_passes_word_timestamps_true(monkeypatch):
    """Verify that the warm-up call to transcribe passes word_timestamps=True."""
    import faster_whisper

    calls = []

    class _TrackerModel:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def transcribe(self, audio_path, **kwargs):
            calls.append(kwargs)
            assert kwargs["word_timestamps"] is True
            word = SimpleNamespace(start=0.0, end=1.0, word=" ", probability=1.0)
            segment = SimpleNamespace(start=0.0, end=1.0, text="", words=[word])
            info = SimpleNamespace(language="en")
            return iter([segment]), info

    monkeypatch.setattr(faster_whisper, "WhisperModel", _TrackerModel)

    host = ModelHost("test-model", device="cpu")
    host.load()

    assert len(calls) == 1
    assert calls[0]["word_timestamps"] is True


# --- a failed CUDA attempt is remembered for the life of the process ---------------------------
#
# A second CUDA attempt after a failed first one hangs forever inside faster-whisper's encode(),
# so ModelHost must not make one (see _CUDA_UNUSABLE in serve/model.py).

CUBLAS_ERROR = "Library cublas64_12.dll is not found or cannot be loaded"


def _fake_whisper(monkeypatch, cuda="fails_warmup"):
    """Install a fake WhisperModel and return the list of devices it was constructed for, in order.

    `cuda` is how a CUDA model behaves: "works", "fails_warmup" (the cuBLAS case: construction
    succeeds, the first transcribe raises) or "fails_construction".
    """
    import faster_whisper

    constructed = []

    class _CountingModel:
        def __init__(self, *args, device=None, compute_type=None, **kwargs):
            constructed.append(device)
            if device == "cuda" and cuda == "fails_construction":
                raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")
            self.device = device

        def transcribe(self, audio, **kwargs):
            if self.device == "cuda" and cuda == "fails_warmup":
                raise RuntimeError(CUBLAS_ERROR)
            word = SimpleNamespace(start=0.0, end=1.0, word=" ", probability=1.0)
            segment = SimpleNamespace(start=0.0, end=1.0, text="", words=[word])
            return iter([segment]), SimpleNamespace(language="en")

    monkeypatch.setattr(faster_whisper, "WhisperModel", _CountingModel)
    return constructed


def test_first_auto_host_falls_back_to_cpu_after_one_cuda_attempt(monkeypatch):
    constructed = _fake_whisper(monkeypatch)

    host = ModelHost("test-model", device="auto")
    host.load()

    assert constructed == ["cuda", "cpu"]
    assert (host.device, host.compute_type, host.loaded) == ("cpu", "int8", True)
    assert model_module._CUDA_UNUSABLE == f"RuntimeError: {CUBLAS_ERROR}"


def test_failed_cuda_attempt_is_logged_once_at_warning(monkeypatch, caplog):
    import logging

    _fake_whisper(monkeypatch)

    with caplog.at_level(logging.INFO, logger=model_module.logger.name):
        ModelHost("test-model", device="auto").load()
        ModelHost("test-model", device="auto").load()  # skips CUDA, must not warn about it again

    disabled = [r for r in caplog.records if "CUDA disabled for this process" in r.getMessage()]
    assert len(disabled) == 1
    assert disabled[0].levelno == logging.WARNING
    assert disabled[0].getMessage() == (
        f"CUDA disabled for this process after: RuntimeError: {CUBLAS_ERROR}; restart the service to retry CUDA"
    )
    skipped = [r for r in caplog.records if "Skipping CUDA" in r.getMessage()]
    assert len(skipped) == 1 and skipped[0].levelno == logging.INFO


def test_second_auto_host_makes_no_cuda_attempt_and_loads_cpu(monkeypatch):
    constructed = _fake_whisper(monkeypatch)
    ModelHost("test-model", device="auto").load()
    assert constructed.count("cuda") == 1
    constructed.clear()

    second = ModelHost("test-model", device="auto")
    second.load()

    assert constructed == ["cpu"]
    assert (second.device, second.compute_type, second.loaded) == ("cpu", "int8", True)


def test_skipping_cuda_keeps_the_requested_cpu_compute_type(monkeypatch):
    constructed = _fake_whisper(monkeypatch)
    ModelHost("test-model", device="auto").load()
    constructed.clear()

    host = ModelHost("test-model", device="auto", compute_type="float32")
    host.load()

    assert constructed == ["cpu"]
    assert (host.device, host.compute_type) == ("cpu", "float32")


def test_explicit_cuda_after_a_failure_raises_without_constructing_a_model(monkeypatch):
    constructed = _fake_whisper(monkeypatch)
    ModelHost("test-model", device="auto").load()
    constructed.clear()

    host = ModelHost("test-model", device="cuda")
    with pytest.raises(ModelLoadError, match="Could not load model 'test-model' on cuda/float16") as excinfo:
        host.load()

    assert constructed == []
    assert host.loaded is False
    assert CUBLAS_ERROR in str(excinfo.value)  # the remembered reason
    assert "restart the service to retry CUDA" in str(excinfo.value)


def test_explicit_cuda_that_fails_is_remembered_too(monkeypatch):
    constructed = _fake_whisper(monkeypatch)

    with pytest.raises(ModelLoadError):
        ModelHost("test-model", device="cuda").load()
    assert constructed == ["cuda"]

    with pytest.raises(ModelLoadError, match="CUDA is disabled for this process"):
        ModelHost("test-model", device="cuda").load()
    assert constructed == ["cuda"]  # the second attempt never built a model


def test_construction_failure_on_cuda_is_remembered_too(monkeypatch):
    constructed = _fake_whisper(monkeypatch, cuda="fails_construction")

    ModelHost("test-model", device="auto").load()
    assert constructed == ["cuda", "cpu"]
    assert model_module._CUDA_UNUSABLE.startswith("RuntimeError: CUDA driver version is insufficient")

    constructed.clear()
    ModelHost("test-model", device="auto").load()
    assert constructed == ["cpu"]


def test_explicit_cpu_is_unaffected_by_a_remembered_cuda_failure(monkeypatch):
    constructed = _fake_whisper(monkeypatch)
    ModelHost("test-model", device="auto").load()
    constructed.clear()

    host = ModelHost("test-model", device="cpu")
    host.load()

    assert constructed == ["cpu"]
    assert (host.device, host.compute_type, host.loaded) == ("cpu", "int8", True)


def test_successful_cuda_load_does_not_set_the_memo(monkeypatch):
    constructed = _fake_whisper(monkeypatch, cuda="works")

    host = ModelHost("test-model", device="auto")
    host.load()

    assert constructed == ["cuda"]
    assert (host.device, host.compute_type) == ("cuda", "float16")
    assert model_module._CUDA_UNUSABLE is None

    # CUDA stays available to later hosts, automatic or explicit.
    ModelHost("test-model", device="auto").load()
    ModelHost("test-model", device="cuda").load()
    assert constructed == ["cuda", "cuda", "cuda"]


def test_a_failing_cpu_load_does_not_set_the_memo(monkeypatch):
    import faster_whisper

    class _Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("weights are corrupt")

    monkeypatch.setattr(faster_whisper, "WhisperModel", _Broken)
    with pytest.raises(ModelLoadError):
        ModelHost("test-model", device="cpu").load()

    assert model_module._CUDA_UNUSABLE is None


def test_reset_cuda_state_lets_cuda_be_tried_again(monkeypatch):
    constructed = _fake_whisper(monkeypatch)
    ModelHost("test-model", device="auto").load()
    assert model_module._CUDA_UNUSABLE is not None

    model_module.reset_cuda_state()

    assert model_module._CUDA_UNUSABLE is None
    constructed.clear()
    ModelHost("test-model", device="auto").load()
    assert constructed == ["cuda", "cpu"]
