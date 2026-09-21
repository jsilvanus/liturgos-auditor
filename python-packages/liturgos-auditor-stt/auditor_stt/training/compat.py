"""Shims for transformers API changes between the versions this package allows."""

from importlib import metadata

from packaging.version import Version


def _transformers_version():
    return Version(metadata.version("transformers"))


def fp32_load_kwargs():
    """from_pretrained kwargs that load fp32 weights.

    transformers 5 defaults to dtype="auto" and would load a checkpoint saved in
    fp16 (as openai's Whisper releases are) as fp16, which breaks fp16 mixed
    precision ("Attempting to unscale FP16 gradients") and trains without fp32
    master weights. `torch_dtype` was renamed `dtype` in 4.56.
    """
    import torch

    name = "dtype" if _transformers_version() >= Version("4.56") else "torch_dtype"
    return {name: torch.float32}


def save_processor(processor, path):
    """Save the tokenizer and feature-extractor files a faster-whisper model directory needs.

    transformers 5 folds the feature-extractor settings into processor_config.json,
    but faster-whisper (mel bin count) and ct2-transformers-converter --copy_files
    read preprocessor_config.json, so write that too.
    """
    processor.save_pretrained(str(path))
    processor.feature_extractor.save_pretrained(str(path))


def warmup_kwargs(ratio):
    """Seq2SeqTrainingArguments kwargs for a warmup fraction (transformers 5 dropped warmup_ratio)."""
    if _transformers_version() >= Version("5"):
        return {"warmup_steps": ratio}
    return {"warmup_ratio": ratio}
