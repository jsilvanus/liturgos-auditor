"""`auditor-stt` CLI: serve the inference server, or run dataset pipeline commands."""

import argparse
import logging
import os
import sys


def _serve(args):
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("auditor_stt.serve.app:app", host="0.0.0.0", port=args.port)


def _dataset_pull(args):
    from .dataset.pull import pull_snapshot

    pull_snapshot(
        base_url=args.base_url,
        corpus_id=args.corpus_id,
        token_env=args.token_env,
        out_dir=args.out,
    )


def _dataset_build(args):
    from .dataset.build import build_dataset

    build_dataset(snapshot_dir=args.snapshot, out_dir=args.out, seed=args.seed)


def _train(args):
    from .training.train import train

    train(
        dataset_dir=args.dataset,
        output_dir=args.out,
        model_id=args.model,
        language=args.language,
        task=args.task,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
    )


def _export(args):
    from .training.export_ct2 import export

    export(model_dir=args.model, output_dir=args.out, quantization=args.quantization)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="auditor-stt")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Run the FastAPI inference server")
    serve_p.add_argument("--port", type=int, default=int(os.environ.get("AUDITOR_STT_PORT", "8090")))
    serve_p.set_defaults(func=_serve)

    dataset_p = sub.add_parser("dataset", help="Dataset pipeline commands")
    dataset_sub = dataset_p.add_subparsers(dest="dataset_command", required=True)

    pull_p = dataset_sub.add_parser("pull", help="Pull a validated snapshot from crowd-source-voice")
    pull_p.add_argument("--base-url", required=True, help="crowd-source-voice base URL, e.g. https://csv.example.org")
    pull_p.add_argument("--corpus-id", required=True, type=int)
    pull_p.add_argument("--token-env", default="CSV_ADMIN_TOKEN", help="Env var holding the admin bearer token")
    pull_p.add_argument("--out", required=True, help="Snapshot output directory")
    pull_p.set_defaults(func=_dataset_pull)

    build_p = dataset_sub.add_parser("build", help="Build an HF dataset + train/dev/test split from a snapshot")
    build_p.add_argument("--snapshot", required=True, help="Snapshot directory produced by `dataset pull`")
    build_p.add_argument("--out", required=True, help="Output directory for the built dataset")
    build_p.add_argument("--seed", type=int, default=42)
    build_p.set_defaults(func=_dataset_build)

    train_p = sub.add_parser("train", help="Fine-tune Whisper on a built dataset")
    train_p.add_argument("--dataset", required=True)
    train_p.add_argument("--out", required=True)
    train_p.add_argument("--model", default="openai/whisper-large-v3-turbo")
    train_p.add_argument("--language", default="fi")
    train_p.add_argument("--task", default="transcribe")
    train_p.add_argument("--learning-rate", type=float, default=1e-5)
    train_p.add_argument("--epochs", type=float, default=3.0)
    train_p.add_argument("--batch-size", type=int, default=4)
    train_p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--fp16", action="store_true")
    train_p.add_argument("--bf16", action="store_true")
    train_p.set_defaults(func=_train)

    export_p = sub.add_parser("export", help="Convert a Transformers Whisper checkpoint to CTranslate2")
    export_p.add_argument("--model", required=True)
    export_p.add_argument("--out", required=True)
    export_p.add_argument("--quantization", default="float16", choices=["float16", "int8_float16", "int8"])
    export_p.set_defaults(func=_export)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
