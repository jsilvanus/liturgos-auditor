"""Export a trained Hugging Face Whisper checkpoint to ONNX for the Aidos Engine.

The Aidos Engine (and so Dictator on a phone) runs Whisper on ONNX Runtime, not on
CTranslate2, so a fine-tune that only exists as a faster-whisper model cannot be used
there. This step is the sibling of `export_ct2`: same input (a full checkpoint, or a
LoRA adapter with `--merge-lora`), different output.

The output directory is a *Whisper package* in the layout the engine's `WhisperPackage`
loader reads (engine/stt in jsilvanus/aidos):

    encoder_model.onnx                  (or encoder_model_quantized.onnx with --quantize int8)
    decoder_model_merged.onnx           (decoder with the KV cache folded in; the engine prefers it)
    tokenizer.json
    generation_config.json              (suppress_tokens)
    preprocessor_config.json            (mel bin count: 80, or 128 for large-v3)
    config.json                         (max_target_positions)

Weights stay out of this public repository: write the output outside the source tree
(`models/` is ignored) and copy it to external storage.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .export_ct2 import _is_adapter, _merge_adapter

ENCODER = "encoder_model"
DECODER = "decoder_model_merged"

# The engine fails to load a package without these; a missing one must fail the export.
REQUIRED_FILES = ("tokenizer.json", "preprocessor_config.json")
# Read when present (suppress tokens, max positions); copied through.
OPTIONAL_FILES = ("generation_config.json", "config.json", "tokenizer_config.json", "special_tokens_map.json")


class PackageError(ValueError):
    """The output directory is not a package the Aidos Engine can load."""


def _onnx_files(directory, stem):
    return sorted(p for p in Path(directory).glob(f"{stem}*.onnx"))


def check_package(directory, mel_bins=None):
    """Raise PackageError unless `directory` is loadable by the engine's WhisperPackage.

    Mirrors engine/stt WhisperPackage.isPackage: tokenizer.json plus one encoder and one decoder graph.
    Returns the number of mel bins declared by preprocessor_config.json.
    """
    directory = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise PackageError(f"{directory} lacks {', '.join(missing)}")
    if not _onnx_files(directory, ENCODER):
        raise PackageError(f"{directory} has no {ENCODER}*.onnx")
    if not _onnx_files(directory, DECODER):
        raise PackageError(
            f"{directory} has no {DECODER}*.onnx (the engine could fall back to decoder_model.onnx, "
            "but that has no KV cache and is far slower; export with the -with-past task)"
        )
    feature_size = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8")).get("feature_size")
    if feature_size not in (80, 128):
        raise PackageError(f"preprocessor_config.json feature_size must be 80 or 128, got {feature_size!r}")
    if mel_bins is not None and feature_size != mel_bins:
        raise PackageError(f"preprocessor_config.json says {feature_size} mel bins, expected {mel_bins}")
    return feature_size


def _optimum_export(optimum, model_dir, output_dir):
    cmd = [
        optimum, "export", "onnx",
        "--model", str(model_dir),
        "--task", "automatic-speech-recognition-with-past",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


def _quantize_int8(directory):
    """Dynamic int8 weights, the phone-sized variant. Replaces the fp32 graphs with *_quantized.onnx."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for stem in (ENCODER, DECODER):
        source = Path(directory) / f"{stem}.onnx"
        if not source.is_file():
            raise PackageError(f"cannot quantize: {source} was not produced by the export")
        target = Path(directory) / f"{stem}_quantized.onnx"
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
        source.unlink()
        # Graphs over 2 GB keep their weights in a sibling file.
        for data in Path(directory).glob(f"{stem}.onnx_data"):
            data.unlink()


def _finish(output_dir, model_dir, quantize):
    """Prune what the engine does not read, copy the tokenizer/config files, verify the layout."""
    output_dir = Path(output_dir)
    # Older optimum writes the unmerged decoders next to the merged one; the engine only uses the merged graph.
    for name in ("decoder_model.onnx", "decoder_with_past_model.onnx"):
        for leftover in (output_dir / name, output_dir / f"{name}_data"):
            if leftover.is_file() and (output_dir / f"{DECODER}.onnx").is_file():
                leftover.unlink()
    for name in (*REQUIRED_FILES, *OPTIONAL_FILES):
        target = output_dir / name
        source = Path(model_dir) / name
        if not target.is_file() and source.is_file():
            shutil.copy2(source, target)
    if quantize == "int8":
        _quantize_int8(output_dir)
    return check_package(output_dir)


def export(model_dir, output_dir, quantize="none", merge_lora=False):
    """Write a Whisper ONNX package for the Aidos Engine to `output_dir` and return it."""
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    if quantize not in ("none", "int8"):
        raise ValueError(f"quantize must be 'none' or 'int8', got {quantize!r}")
    adapter = _is_adapter(model_dir)
    if adapter and not merge_lora:
        raise ValueError(
            f"{model_dir} is a LoRA adapter, not a full model; "
            "pass --merge-lora to merge it into its base model before converting"
        )
    optimum = shutil.which("optimum-cli")
    if optimum is None:
        raise RuntimeError("optimum-cli is not installed; install the onnx extra (pip install \".[onnx]\")")
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a new output directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if not adapter:
        _optimum_export(optimum, model_dir, output_dir)
        _finish(output_dir, model_dir, quantize)
        return output_dir

    merged_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-merged-", dir=output_dir.parent))
    try:
        _merge_adapter(model_dir, merged_dir)
        _optimum_export(optimum, merged_dir, output_dir)
        _finish(output_dir, merged_dir, quantize)
    finally:
        shutil.rmtree(merged_dir, ignore_errors=True)
    return output_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="Export a trained Whisper model as an Aidos Engine ONNX package")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quantize", default="none", choices=["none", "int8"],
                   help="int8 replaces the fp32 graphs with dynamic-int8 ones (about a quarter of the size; "
                        "re-run `eval --backend onnx` on it, int8 costs some accuracy)")
    p.add_argument("--merge-lora", action="store_true")
    args = p.parse_args(argv)
    export(args.model, args.output, args.quantize, args.merge_lora)


if __name__ == "__main__":
    main()
