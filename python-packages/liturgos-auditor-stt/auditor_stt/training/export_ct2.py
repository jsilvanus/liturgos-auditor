"""Convert a trained Hugging Face Whisper checkpoint to CTranslate2."""

import argparse
import shutil
import subprocess
from pathlib import Path


def export(model_dir, output_dir, quantization="float16"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = shutil.which("ct2-transformers-converter")
    if converter is None:
        raise RuntimeError("ct2-transformers-converter is not installed; install the training dependencies")
    cmd = [
        converter,
        "--model", str(model_dir),
        "--output_dir", str(output_dir),
        "--copy_files", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "--quantization", quantization,
    ]
    subprocess.run(cmd, check=True)
    return output_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="Export a trained Whisper model for faster-whisper")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quantization", default="float16",
                   choices=["float16", "int8_float16", "int8"])
    args = p.parse_args(argv)
    export(args.model, args.output, args.quantization)


if __name__ == "__main__":
    main()
