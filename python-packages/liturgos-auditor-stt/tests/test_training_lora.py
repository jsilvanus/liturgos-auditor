"""Torch-dependent training tests on a tiny randomly initialised Whisper: no downloads."""

import io
import types
import wave
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    import peft  # noqa: F401
    import transformers
    from datasets import Dataset, DatasetDict
    from transformers import WhisperConfig, WhisperForConditionalGeneration
except Exception as exc:  # ImportError, or OSError when a host policy blocks the torch DLLs
    pytest.skip(f"training dependencies unavailable: {exc}", allow_module_level=True)

from auditor_stt.training import compat, export_ct2, preflight
from auditor_stt.training import train as train_module


def _tiny_config():
    return WhisperConfig(
        vocab_size=100, num_mel_bins=8, d_model=16, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_source_positions=30, max_target_positions=20, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, decoder_start_token_id=3,
    )


def _tiny_model():
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(_tiny_config())
    model.config.use_cache = False
    return model


def _forward(model, seed=1):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(2, 8, 60, generator=generator)
    labels = torch.randint(4, 100, (2, 5), generator=generator)
    return model(input_features=features, labels=labels)


def test_lora_makes_only_the_adapters_trainable_and_keeps_them_fp32():
    model = _tiny_model().half()  # a half-precision base, as fp16 training would use
    peft_model = train_module._apply_lora(model, 4, 8, 0.0, ("q_proj", "v_proj"), gradient_checkpointing=False)

    trainable = {n: p for n, p in peft_model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())
    count, total = peft_model.get_nb_trainable_parameters()
    assert 0 < count < total / 2


def test_lora_target_modules_are_configurable():
    peft_model = train_module._apply_lora(_tiny_model(), 4, 8, 0.0, ("k_proj",), gradient_checkpointing=False)
    names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert names and all(".k_proj." in n for n in names)


def test_lora_gradients_reach_the_encoder_with_reentrant_gradient_checkpointing():
    # Reentrant checkpointing needs an input that requires grad; the frozen mel
    # features do not, so without enable_input_require_grads nothing would train.
    peft_model = train_module._apply_lora(_tiny_model(), 4, 8, 0.0, ("q_proj", "v_proj"), gradient_checkpointing=True)
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    peft_model.train()

    _forward(peft_model).loss.backward()

    # lora_B starts at zero, so lora_A gets no gradient until B moves: check B.
    encoder_b = [p for n, p in peft_model.named_parameters() if ".encoder.layers" in n and "lora_B" in n]
    assert encoder_b and all(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder_b)


def test_lora_model_matches_base_before_training_then_merge_matches_the_adapter(tmp_path, monkeypatch):
    base_dir = tmp_path / "base"
    _tiny_model().save_pretrained(base_dir)
    base = WhisperForConditionalGeneration.from_pretrained(base_dir)
    peft_model = train_module._apply_lora(base, 4, 8, 0.0, ("q_proj", "v_proj"), gradient_checkpointing=False)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.add_(0.05)  # pretend training moved the adapters
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    # what train.py leaves next to the adapter (forced language and task)
    transformers.GenerationConfig(language="fi", task="transcribe").save_pretrained(adapter_dir)

    class _Processor:
        @staticmethod
        def from_pretrained(source):
            return _Processor()

        feature_extractor = types.SimpleNamespace(
            save_pretrained=lambda path: (Path(path) / "preprocessor_config.json").write_text("{}"))

        def save_pretrained(self, path):  # transformers 5 writes processor_config.json instead
            (Path(path) / "processor_config.json").write_text("{}")

    monkeypatch.setattr(transformers, "WhisperProcessor", _Processor)
    (tmp_path / "merged").mkdir()
    merged_dir = export_ct2._merge_adapter(adapter_dir, tmp_path / "merged")

    merged = WhisperForConditionalGeneration.from_pretrained(merged_dir)
    assert not any("lora" in name for name, _ in merged.named_parameters())
    assert merged.generation_config.language == "fi"
    assert (merged_dir / "preprocessor_config.json").exists()
    peft_model.eval()
    merged.eval()
    with torch.no_grad():
        expected = _forward(peft_model).logits
        actual = _forward(merged).logits
    assert torch.allclose(expected, actual, atol=1e-5)
    # the adapter really changed the output, so this compared something
    reference = WhisperForConditionalGeneration.from_pretrained(base_dir).eval()
    with torch.no_grad():
        assert not torch.allclose(_forward(reference).logits, actual, atol=1e-5)


def test_fp32_load_kwargs_ask_for_float32():
    (value,) = compat.fp32_load_kwargs().values()
    assert value is torch.float32


def test_count_parameters_falls_back_to_the_config_without_loading_weights(tmp_path):
    _tiny_config().save_pretrained(tmp_path)
    expected = sum(p.numel() for p in WhisperForConditionalGeneration(_tiny_config()).parameters())
    assert preflight.count_parameters(str(tmp_path)) == expected


def test_audio_array_decodes_bytes_and_resamples(tmp_path):
    rate = 8000
    samples = (np.sin(2 * np.pi * 440 * np.arange(rate) / rate) * 20000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    (tmp_path / "tone.wav").write_bytes(buffer.getvalue())

    from_bytes = train_module.audio_array({"bytes": buffer.getvalue(), "path": None}, 16000)
    from_path = train_module.audio_array({"bytes": None, "path": str(tmp_path / "tone.wav")}, 16000)

    assert from_bytes.dtype == np.float32
    assert abs(len(from_bytes) - 16000) <= 1  # 1 s at 8 kHz became 1 s at 16 kHz
    assert 0.5 < np.abs(from_bytes).max() <= 1.0
    np.testing.assert_allclose(from_bytes, from_path, atol=1e-6)


# --- train() runs the preflight before touching the model ------------------------------

class _ModelTouched(Exception):
    pass


@pytest.fixture
def dataset_dir(tmp_path):
    split = Dataset.from_dict({"text": ["x"], "duration": [1.0]})
    DatasetDict({"train": split, "dev": split, "test": split}).save_to_disk(str(tmp_path / "ds"))
    return tmp_path / "ds"


@pytest.fixture
def stop_at_model_load(monkeypatch):
    def load(*args, **kwargs):
        raise _ModelTouched

    monkeypatch.setattr(train_module.WhisperProcessor, "from_pretrained", load)


def test_train_runs_the_preflight_before_loading_anything(tmp_path, dataset_dir, stop_at_model_load, monkeypatch):
    calls = []

    def check(*args):
        calls.append(args)
        raise preflight.PreflightError("does not fit")

    monkeypatch.setattr(train_module, "check_gpu_memory", check)

    with pytest.raises(preflight.PreflightError, match="does not fit"):
        train_module.train(dataset_dir, tmp_path / "out", preset="small", peft="lora", fp16=True,
                           gradient_checkpointing=True, batch_size=2)

    # resolved model id, mode, half precision, checkpointing, batch size
    assert calls == [("openai/whisper-small", "lora", True, True, 2)]


def test_train_uses_bf16_as_half_precision_and_defaults_to_full_finetuning_of_turbo(
        tmp_path, dataset_dir, stop_at_model_load, monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "check_gpu_memory", lambda *args: calls.append(args))

    with pytest.raises(_ModelTouched):
        train_module.train(dataset_dir, tmp_path / "out", bf16=True)

    assert calls == [("openai/whisper-large-v3-turbo", "full", True, False, 4)]


def test_skip_preflight_bypasses_the_check(tmp_path, dataset_dir, stop_at_model_load, monkeypatch):
    def check(*args):
        raise AssertionError("preflight must not run")

    monkeypatch.setattr(train_module, "check_gpu_memory", check)

    with pytest.raises(_ModelTouched):
        train_module.train(dataset_dir, tmp_path / "out", skip_preflight=True)


def test_explicit_model_id_wins_over_the_preset(tmp_path, dataset_dir, stop_at_model_load, monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "check_gpu_memory", lambda *args: calls.append(args))

    with pytest.raises(_ModelTouched):
        train_module.train(dataset_dir, tmp_path / "out", model_id="org/own-whisper", preset="small")

    assert calls[0][0] == "org/own-whisper"


def test_train_rejects_unknown_peft_and_bad_dataset_before_anything_else(tmp_path, dataset_dir):
    with pytest.raises(ValueError, match="peft"):
        train_module.train(dataset_dir, tmp_path / "out", peft="qlora")

    DatasetDict({"train": Dataset.from_dict({"text": ["x"]})}).save_to_disk(str(tmp_path / "bad"))
    with pytest.raises(ValueError, match="train, dev and test"):
        train_module.train(tmp_path / "bad", tmp_path / "out")
