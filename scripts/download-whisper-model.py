#!/usr/bin/env python3
"""Download and locally validate the configured faster-whisper model."""

from __future__ import annotations

import argparse
import os


DEFAULT_MODEL = "Systran/faster-whisper-large-v3-turbo"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a faster-whisper model into the local cache."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=os.environ.get("AUDITOR_STT_MODEL", DEFAULT_MODEL),
        help=f"Model ID (default: AUDITOR_STT_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("AUDITOR_STT_MODEL_DIR"),
        help="Model cache/download directory (default: AUDITOR_STT_MODEL_DIR or faster-whisper default)",
    )
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    print(f"Downloading/loading model: {args.model}")
    if args.model_dir:
        print(f"Model directory: {args.model_dir}")

    # CPU/int8 keeps this command usable on machines without CUDA. The
    # constructor downloads the model if it is not already present.
    WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        download_root=args.model_dir,
    )

    print("Model is available locally.")


if __name__ == "__main__":
    main()
