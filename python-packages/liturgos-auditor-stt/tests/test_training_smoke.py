"""Mechanics smoke test: train -> export on a real tiny Whisper, with and without LoRA.

Checks that the pieces work together, not that the model learns anything. It
downloads openai/whisper-tiny (~150 MB of public weights from huggingface.co;
no dataset content leaves the machine), so it only runs on request:

    AUDITOR_STT_RUN_SLOW=1 pytest tests/test_training_smoke.py
"""

import json
import os
import shutil
import wave

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("AUDITOR_STT_RUN_SLOW") != "1",
        reason="set AUDITOR_STT_RUN_SLOW=1 to run (downloads openai/whisper-tiny)",
    ),
]

try:
    import torch  # noqa: F401
    import peft  # noqa: F401
    from datasets import Audio, Dataset, DatasetDict
except Exception as exc:  # ImportError, or OSError when a host policy blocks the torch DLLs
    pytest.skip(f"training dependencies unavailable: {exc}", allow_module_level=True)

from auditor_stt.training.export_ct2 import export
from auditor_stt.training.lineage import sha256_file
from auditor_stt.training.train import train

SAMPLE_RATE = 16000


def _write_wav(path, rng, seconds, kind):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    if kind == "sine":
        signal = 0.4 * np.sin(2 * np.pi * rng.uniform(200, 800) * t)
    else:
        signal = 0.1 * rng.standard_normal(len(t))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((signal * 32767).astype("<i2").tobytes())


def _synthetic_dataset(root, sizes=(24, 4, 4)):
    """A few dozen 1-2 s sine/noise clips with dummy Finnish text, split like `dataset build` output."""
    rng = np.random.default_rng(0)
    audio_dir = root / "audio"
    audio_dir.mkdir()
    splits = {}
    n = 0
    for name, size in zip(("train", "dev", "test"), sizes):
        rows = []
        for _ in range(size):
            path = audio_dir / f"{n:04d}.wav"
            seconds = float(rng.uniform(1.0, 2.0))
            _write_wav(path, rng, seconds, "sine" if n % 2 == 0 else "noise")
            rows.append({"audio": str(path), "text": f"Tämä on testilause numero {n}.", "duration": seconds})
            n += 1
        splits[name] = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    dataset_dir = root / "dataset"
    DatasetDict(splits).save_to_disk(str(dataset_dir))
    (dataset_dir / "build_metadata.json").write_text(json.dumps({"dataset_version": "smoke-v1"}))
    (dataset_dir / "manifest.json").write_text(json.dumps({"recordings": list(range(n))}))
    return dataset_dir


@pytest.mark.parametrize("peft_mode", ["lora", "none"])
def test_train_then_export_on_whisper_tiny(tmp_path, peft_mode):
    lora = peft_mode == "lora"
    dataset_dir = _synthetic_dataset(tmp_path)
    out = tmp_path / "run"

    metadata = train(
        dataset_dir, out, model_id="openai/whisper-tiny", peft=peft_mode, lora_r=8, lora_alpha=16,
        epochs=1, batch_size=8, learning_rate=1e-3,
    )

    # LoRA saves an adapter, full fine-tuning a whole model
    assert (out / "adapter_config.json").is_file() == lora
    assert (out / "adapter_model.safetensors").is_file() == lora
    assert (out / "model.safetensors").is_file() != lora
    assert (out / "preprocessor_config.json").is_file()
    assert (out / "tokenizer.json").is_file()
    assert (out / "generation_config.json").is_file()

    saved = json.loads((out / "training_metadata.json").read_text())
    assert saved["base_model"] == "openai/whisper-tiny"
    assert (saved["train_examples"], saved["dev_examples"], saved["test_examples"]) == (24, 4, 4)
    assert saved["dataset_version"] == "smoke-v1"
    assert saved["dataset_manifest_sha256"] == sha256_file(dataset_dir / "manifest.json")
    assert {"base_model_revision", "git_commit", "package_versions", "config", "peft", "preset"} <= set(saved)
    assert saved["config"]["batch_size"] == 8 and saved["package_versions"]["peft"]
    assert saved["train_metrics"] and metadata["dataset_version"] == "smoke-v1"
    assert saved["peft"]["method"] == peft_mode
    if lora:
        assert saved["peft"]["r"] == 8
        assert 0 < saved["peft"]["trainable_parameters"] < saved["peft"]["total_parameters"]
        # an adapter must be merged explicitly
        with pytest.raises(ValueError, match="--merge-lora"):
            export(out, tmp_path / "ct2-refused")

    if shutil.which("ct2-transformers-converter") is None:
        pytest.skip("ct2-transformers-converter not installed; export not exercised")
    ct2 = export(out, tmp_path / "ct2", quantization="int8", merge_lora=lora)
    for name in ("model.bin", "config.json", "preprocessor_config.json", "tokenizer.json"):
        assert (ct2 / name).is_file(), name
    assert (ct2 / "_merged_hf" / "config.json").is_file() == lora
    if lora:
        assert not any("lora" in p.name for p in (ct2 / "_merged_hf").iterdir())

    # the shipping loader accepts the result
    from faster_whisper import WhisperModel

    model = WhisperModel(str(ct2), device="cpu", compute_type="int8")
    audio = np.random.default_rng(1).standard_normal(SAMPLE_RATE).astype("float32") * 0.05
    segments, _ = model.transcribe(audio, language="fi")
    list(segments)  # decoding runs; the text of an untrained tiny model is meaningless
