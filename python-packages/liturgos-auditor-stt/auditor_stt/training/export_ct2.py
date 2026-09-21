"""Convert a trained Hugging Face Whisper checkpoint to CTranslate2.

A LoRA adapter directory (from `train --peft lora`) is not a model: it is merged
into its base model first, and the merged checkpoint is kept in `_merged_hf`
next to the CTranslate2 files.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def _is_adapter(model_dir):
    return (Path(model_dir) / "adapter_config.json").is_file()


def _merge_adapter(adapter_dir, merged_dir):
    """Fold the adapter into its base model and save a plain HF checkpoint with processor files.

    ct2-transformers-converter only understands ordinary checkpoints.
    """
    from peft import PeftModel
    from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

    from .compat import fp32_load_kwargs, save_processor

    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_id = config.get("base_model_name_or_path")
    if not base_id:
        raise ValueError(f"{adapter_dir / 'adapter_config.json'} does not name its base model")

    base = WhisperForConditionalGeneration.from_pretrained(base_id, **fp32_load_kwargs())
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    if (adapter_dir / "generation_config.json").is_file():
        model.generation_config = GenerationConfig.from_pretrained(str(adapter_dir))
    model.save_pretrained(str(merged_dir))
    # train.py saves the processor next to the adapter; fall back to the base model's.
    processor_source = adapter_dir if (adapter_dir / "preprocessor_config.json").is_file() else base_id
    save_processor(WhisperProcessor.from_pretrained(str(processor_source)), merged_dir)
    return Path(merged_dir)


# faster-whisper needs these next to model.bin (mel bin count, offline tokenizer): a
# missing one must fail the conversion, not be skipped. The others are optional, and
# transformers 5 no longer writes special_tokens_map.json.
REQUIRED_COPY_FILES = ("preprocessor_config.json", "tokenizer.json")
OPTIONAL_COPY_FILES = ("tokenizer_config.json", "special_tokens_map.json")


def _convert(converter, model_dir, output_dir, quantization):
    copy_files = [*REQUIRED_COPY_FILES, *(f for f in OPTIONAL_COPY_FILES if (Path(model_dir) / f).is_file())]
    cmd = [
        converter,
        "--model", str(model_dir),
        "--output_dir", str(output_dir),
        "--copy_files", *copy_files,
        "--quantization", quantization,
    ]
    subprocess.run(cmd, check=True)


def export(model_dir, output_dir, quantization="float16", merge_lora=False):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    adapter = _is_adapter(model_dir)
    if adapter and not merge_lora:
        raise ValueError(
            f"{model_dir} is a LoRA adapter, not a full model; "
            "pass --merge-lora to merge it into its base model before converting"
        )
    converter = shutil.which("ct2-transformers-converter")
    if converter is None:
        raise RuntimeError("ct2-transformers-converter is not installed; install the training dependencies")
    # The converter refuses an existing output directory (and --force would delete it).
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a new output directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if not adapter:
        _convert(converter, model_dir, output_dir, quantization)
        return output_dir

    # Merge beside the output, convert, then move the merged checkpoint inside it,
    # so the converter gets the fresh output directory it insists on.
    merged_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-merged-", dir=output_dir.parent))
    try:
        _merge_adapter(model_dir, merged_dir)
        _convert(converter, merged_dir, output_dir, quantization)
        shutil.move(str(merged_dir), str(output_dir / "_merged_hf"))
    finally:
        shutil.rmtree(merged_dir, ignore_errors=True)
    return output_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="Export a trained Whisper model for faster-whisper")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quantization", default="float16",
                   choices=["float16", "int8_float16", "int8"])
    p.add_argument("--merge-lora", action="store_true",
                   help="Merge a LoRA adapter into its base model first (required for adapter directories)")
    args = p.parse_args(argv)
    export(args.model, args.output, args.quantization, args.merge_lora)


if __name__ == "__main__":
    main()
