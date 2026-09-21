"""`auditor-stt` CLI: serve the inference server, or run dataset pipeline commands."""

import argparse
import logging
import os
import sys


def _serve(args):
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    # Batch jobs need a data dir; a local `serve` gets ./data unless one is configured.
    os.environ.setdefault("AUDITOR_STT_DATA_DIR", "./data")
    uvicorn.run("auditor_stt.serve.app:app", host="0.0.0.0", port=args.port)


def _dataset_pull(args):
    from .dataset.pull import pull_snapshot

    pull_snapshot(
        base_url=args.base_url,
        corpus_id=args.corpus_id,
        token_env=args.token_env,
        out_dir=args.out,
    )


def _dataset_sync(args):
    import httpx

    from .dataset.pull import ExportInconsistentError, UnsupportedCorpusTypeError
    from .dataset.sync import MassRemovalError, run_sync

    try:
        report = run_sync(
            base_url=args.base_url,
            corpus_id=args.corpus_id,
            data_dir=args.data_dir,
            token_env=args.token_env,
            speaker_salt_env=args.speaker_salt_env,
            allow_mass_removal=args.allow_mass_removal,
        )
    except (UnsupportedCorpusTypeError, ExportInconsistentError, MassRemovalError, RuntimeError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Corpus {args.corpus_id}: {report.summary()}")
    return 0


def _resolve_data_dir(args):
    return args.data_dir or os.environ.get("AUDITOR_STT_TRAIN_DATA_DIR", "./data")


def _dataset_build(args):
    from .dataset.build import build_dataset, build_from_ledger, datasets_root
    from .dataset.ledger import LedgerError

    logging.basicConfig(level=logging.INFO)
    if args.snapshot:
        if args.out is None or args.data_dir is not None or args.corpus_id is not None:
            print("error: --snapshot needs --out and cannot be combined with --data-dir or --corpus-id", file=sys.stderr)
            return 2

    # Validate max_drop_fraction if provided.
    max_drop_frac = None
    if args.max_drop_fraction is not None:
        if not (0 < args.max_drop_fraction <= 1):
            print(f"error: --max-drop-fraction must be between 0 (exclusive) and 1 (inclusive), got {args.max_drop_fraction}", file=sys.stderr)
            return 2
        max_drop_frac = args.max_drop_fraction

    try:
        if args.snapshot:
            metadata = build_dataset(args.snapshot, args.out, seed=args.seed, force=args.force, max_drop_fraction=max_drop_frac)
            location = args.out
        else:
            data_dir = _resolve_data_dir(args)
            metadata = build_from_ledger(
                data_dir, out_root=args.out, corpus_id=args.corpus_id, seed=args.seed, force=args.force, max_drop_fraction=max_drop_frac
            )
            location = os.path.join(args.out or datasets_root(data_dir), metadata["dataset_version"])
    except (ValueError, OSError, LedgerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sizes = metadata["split_sizes"]
    print(
        f"Dataset {metadata['dataset_version']}: train {sizes['train']}, dev {sizes['dev']}, test {sizes['test']} "
        f"({metadata['train_rows_dropped_for_transcript_overlap']} dropped for transcript overlap, "
        f"{metadata.get('train_speakers_lost', 0)} speakers lost) -> {location}"
    )
    return 0


def _print_lines(lines):
    print("\n".join(lines))


def _dataset_lineage(args):
    import json

    from .dataset.lineage import render_lineage, speaker_lineage

    report = speaker_lineage(_resolve_data_dir(args), args.speaker, models_dir=args.models_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_lines(render_lineage(report))
    return 0


def _dataset_purge(args):
    from .dataset.lineage import purge_speaker, render_purge

    try:
        report = purge_speaker(_resolve_data_dir(args), args.speaker, models_dir=args.models_dir, apply=args.yes)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_lines(render_purge(report))
    return 0


def _dataset_prune(args):
    from .dataset.lineage import prune, render_prune

    if args.keep_datasets is None and args.runs_dir is None:
        print("error: nothing to prune; pass --keep-datasets N and/or --runs-dir DIR", file=sys.stderr)
        return 2
    if args.keep_datasets is not None and args.keep_datasets < 0:
        print("error: --keep-datasets must not be negative", file=sys.stderr)
        return 2
    try:
        report = prune(
            _resolve_data_dir(args), keep_datasets=args.keep_datasets, models_dir=args.models_dir,
            include_referenced=args.include_referenced, runs_dir=args.runs_dir, apply=args.yes,
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_lines(render_prune(report))
    return 0


def _train(args):
    from .training.preflight import PreflightError  # stdlib only, unlike training.train
    from .training.train import train

    logging.basicConfig(level=logging.INFO)
    try:
        train(
            dataset_dir=args.dataset,
            output_dir=args.out,
            model_id=args.model,
            preset=args.preset,
            language=args.language,
            task=args.task,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            fp16=args.fp16,
            bf16=args.bf16,
            peft=args.peft,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            gradient_checkpointing=args.gradient_checkpointing,
            skip_preflight=args.skip_preflight,
        )
    except PreflightError as exc:
        # The refusal carries the advice (what to change and roughly what it saves): show it, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _export(args):
    import subprocess

    from .training.export_ct2 import export

    try:
        export(model_dir=args.model, output_dir=args.out, quantization=args.quantization, merge_lora=args.merge_lora)
    except (ValueError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as exc:
        # export() raises these deliberately (adapter without --merge-lora, existing output, missing converter).
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _eval(args):
    """Exit code 0: gate passed, 3: gate failed, 1: could not evaluate. Prints metrics, never transcript text."""
    import json
    from pathlib import Path

    from .serve.audio import MediaDecodeError
    from .serve.model import AudioDecodeError, ModelLoadError
    from .training.gate import evaluate_gate, gate_path, render_gate

    logging.basicConfig(level=logging.INFO)
    predictions = [] if args.dump_predictions else None
    try:
        if predictions is not None:  # fail on an unwritable path now, not after the evaluation
            Path(args.dump_predictions).parent.mkdir(parents=True, exist_ok=True)
        gate = evaluate_gate(
            args.model,
            args.dataset,
            split=args.split,
            baseline=args.baseline,
            backend=args.backend,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            longform_audio=args.longform_audio,
            longform_reference=args.longform_reference,
            longform_tolerance=args.longform_tolerance,
            min_improvement=args.min_improvement,
            require_longform=args.require_longform,
            out=args.out,
            predictions=predictions,
        )
        if predictions is not None:
            with open(args.dump_predictions, "w", encoding="utf-8") as f:
                for row in predictions:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (ValueError, OSError, RuntimeError, ImportError, ModelLoadError, MediaDecodeError, AudioDecodeError) as exc:
        cause = f" ({exc.__cause__})" if exc.__cause__ else ""
        print(f"error: {exc}{cause}", file=sys.stderr)
        return 1
    _print_lines(render_gate(gate))
    print(f"Gate written to {gate_path(args.model, args.out)}")
    if predictions is not None:
        print(f"Predictions, including transcript text, written to {args.dump_predictions}")
    return 0 if gate["passed"] else 3


def _registry_dir(args):
    from .serve.registry import default_registry_dir

    return args.registry_dir or default_registry_dir()


def _models_register(args):
    from .serve.registry import RegistryError, register

    try:
        metadata = register(
            _registry_dir(args), args.name, args.version, args.export,
            training_run_dir=args.training_run, gate_path=args.gate, quantization=args.quantization, move=args.move,
        )
    except (RegistryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    gate = metadata["gate"]
    verdict = "no gate" if gate is None else "gate passed" if gate["passed"] else "gate NOT passed"
    print(f"Registered {args.name}@{args.version} (dataset_version {metadata['dataset_version'] or 'unknown'}, {verdict})")
    return 0


def _models_promote(args):
    from .serve.registry import GateRefusedError, RegistryError, gate_summary, promote, read_gate

    registry_dir = _registry_dir(args)
    try:
        _print_lines(gate_summary(read_gate(registry_dir, args.name, args.version)))
        current = promote(registry_dir, args.name, args.version, force=args.force)
    except GateRefusedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (RegistryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    forced = " (forced past the gate)" if current["forced"] else ""
    print(f"{args.name}: current version is now {current['version']}{forced}")
    return 0


def _models_list(args):
    from .serve.registry import list_models

    try:
        models = list_models(_registry_dir(args))
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not models:
        print("No models registered")
    for model in models:
        print(f"{model['name']}  current: {model['current_version'] or 'none'}  versions: {', '.join(model['versions'])}")
    return 0


def _models_show(args):
    from .serve.registry import RegistryError, gate_summary, show

    try:
        details = show(_registry_dir(args), args.name, args.version)
    except (RegistryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    current = details["current"]
    print(f"Model {details['name']}")
    print(f"Versions: {', '.join(details['versions'])}")
    if current:
        forced = " (forced past the gate)" if current.get("forced") else ""
        print(f"Current: {current.get('version')}, promoted {current.get('promoted_at')} by {current.get('promoted_by')}{forced}")
    else:
        print("Current: none promoted yet")
    if details["version"] is None:
        return 0
    print(f"Version {details['version']}:")
    for key in ("created_at", "dataset_version", "base_model", "quantization"):
        value = (details["metadata"] or {}).get(key)
        if value is not None:
            print(f"  {key}: {value}")
    _print_lines(f"  {line}" for line in gate_summary(details["gate"]))
    return 0


def main(argv=None):
    from .training.preflight import PRESETS  # stdlib only, unlike the rest of training/

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

    sync_p = dataset_sub.add_parser("sync", help="Sync a crowd-source-voice corpus into the local ledger (incremental)")
    sync_p.add_argument("--base-url", required=True, help="crowd-source-voice base URL, e.g. https://csv.example.org")
    sync_p.add_argument("--corpus-id", required=True, type=int)
    sync_p.add_argument("--token-env", default="CSV_ADMIN_TOKEN", help="Env var holding the admin or export bearer token")
    sync_p.add_argument(
        "--data-dir",
        default=os.environ.get("AUDITOR_STT_TRAIN_DATA_DIR", "./data"),
        help="Directory holding ledger.sqlite and audio/ (env AUDITOR_STT_TRAIN_DATA_DIR)",
    )
    sync_p.add_argument(
        "--speaker-salt-env",
        default="AUDITOR_STT_SPEAKER_SALT",
        help="Env var holding the salt used to pseudonymise raw csv user ids (without it they are stored speaker-less)",
    )
    sync_p.add_argument(
        "--allow-mass-removal",
        action="store_true",
        help="Permit tombstoning more than half of a corpus's recordings in one sync",
    )
    sync_p.set_defaults(func=_dataset_sync)

    data_dir_help = "Directory holding ledger.sqlite, audio/ and datasets/ (default: env AUDITOR_STT_TRAIN_DATA_DIR or ./data)"
    models_dir_help = "Directory searched for training_metadata.json / metadata.json that name dataset versions"

    build_p = dataset_sub.add_parser(
        "build", help="Build a versioned HF dataset (stable speaker-disjoint split) from the ledger or a snapshot"
    )
    build_p.add_argument("--data-dir", default=None, help=data_dir_help)
    build_p.add_argument("--corpus-id", type=int, default=None, help="Only use this corpus's recordings (ledger mode)")
    build_p.add_argument(
        "--out", default=None,
        help="Ledger mode: parent of the <version> directory (default: <data-dir>/datasets). "
             "Snapshot mode: the output directory itself (required)",
    )
    build_p.add_argument("--snapshot", default=None, help="Build from a `dataset pull` snapshot directory instead of the ledger")
    build_p.add_argument("--seed", type=int, default=42, help="Part of the split salt; keep it to keep the split")
    build_p.add_argument("--force", action="store_true", help="Rebuild even if this dataset version already exists")
    build_p.add_argument("--max-drop-fraction", type=float, default=None,
                        help="Fail the build if the transcript-leakage guard drops more than this fraction of train rows (0 < F <= 1)")
    build_p.set_defaults(func=_dataset_build)

    lineage_p = dataset_sub.add_parser("lineage", help="List the dataset versions and models that used a speaker")
    lineage_p.add_argument("--data-dir", default=None, help=data_dir_help)
    lineage_p.add_argument("--speaker", required=True, help="Pseudonymous speaker id")
    lineage_p.add_argument("--models-dir", default="./models", help=models_dir_help)
    lineage_p.add_argument("--json", action="store_true", help="Machine-readable output")
    lineage_p.set_defaults(func=_dataset_lineage)

    purge_p = dataset_sub.add_parser(
        "purge",
        help="Erase a speaker from the ledger, audio cache and dataset versions (dry run unless --yes); "
             "affected models are only listed. Delete upstream first or the next sync restores the recordings",
    )
    purge_p.add_argument("--data-dir", default=None, help=data_dir_help)
    purge_p.add_argument("--speaker", required=True, help="Pseudonymous speaker id")
    purge_p.add_argument("--models-dir", default="./models", help=models_dir_help)
    purge_p.add_argument("--yes", action="store_true", help="Apply the purge (default: only show what would happen)")
    purge_p.set_defaults(func=_dataset_purge)

    prune_p = dataset_sub.add_parser(
        "prune", help="Retention: delete old dataset versions and intermediate training checkpoints (dry run unless --yes)"
    )
    prune_p.add_argument("--data-dir", default=None, help=data_dir_help)
    prune_p.add_argument("--keep-datasets", type=int, default=None, help="Keep only the newest N dataset versions")
    prune_p.add_argument("--models-dir", default="./models", help=models_dir_help + " (versions they name are kept)")
    prune_p.add_argument(
        "--include-referenced", action="store_true",
        help="Also delete dataset versions that a model still names (loses the record of what it was trained on)",
    )
    prune_p.add_argument("--runs-dir", default=None, help="Delete checkpoint-N directories inside finished training runs under DIR")
    prune_p.add_argument("--yes", action="store_true", help="Apply (default: only show what would happen)")
    prune_p.set_defaults(func=_dataset_prune)

    train_p = sub.add_parser("train", help="Fine-tune Whisper on a built dataset")
    train_p.add_argument("--dataset", required=True)
    train_p.add_argument("--out", required=True)
    train_p.add_argument("--model", default=None,
                         help="Base model id or path; overrides --preset (default: preset large-v3-turbo)")
    train_p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                         help="Base model preset: small suits GPUs with little memory")
    train_p.add_argument("--language", default="fi")
    train_p.add_argument("--task", default="transcribe")
    train_p.add_argument("--learning-rate", type=float, default=1e-5)
    train_p.add_argument("--epochs", type=float, default=3.0)
    train_p.add_argument("--batch-size", type=int, default=4)
    train_p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--fp16", action="store_true")
    train_p.add_argument("--bf16", action="store_true")
    train_p.add_argument("--peft", choices=["none", "lora"], default="none",
                         help="lora trains small adapters instead of all weights (needs far less GPU memory)")
    train_p.add_argument("--lora-r", type=int, default=32)
    train_p.add_argument("--lora-alpha", type=int, default=64)
    train_p.add_argument("--lora-dropout", type=float, default=0.05)
    train_p.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"],
                         help="Space-separated module names to adapt")
    train_p.add_argument("--gradient-checkpointing", action="store_true",
                         help="Trade compute for memory by recomputing activations")
    train_p.add_argument("--skip-preflight", action="store_true",
                         help="Skip the GPU memory estimate that fails early on runs that will not fit")
    train_p.set_defaults(func=_train)

    export_p = sub.add_parser("export", help="Convert a Transformers Whisper checkpoint to CTranslate2")
    export_p.add_argument("--model", required=True)
    export_p.add_argument("--out", required=True)
    export_p.add_argument("--quantization", default="float16", choices=["float16", "int8_float16", "int8"])
    export_p.add_argument("--merge-lora", action="store_true",
                          help="Merge a LoRA adapter into its base model first (required for adapter directories)")
    export_p.set_defaults(func=_export)

    eval_p = sub.add_parser(
        "eval",
        help="Evaluation gate: score a model against the zero-shot baseline on the test split and write gate.json "
             "(exit code 0 passed, 3 failed, 1 error)",
    )
    eval_p.add_argument("--model", required=True,
                        help="Candidate: a CTranslate2 model directory (a Hugging Face checkpoint with --backend hf)")
    eval_p.add_argument("--backend", choices=["ct2", "hf"], default="ct2",
                        help="ct2 scores the exported model through the serving code (default); "
                             "hf scores a merged Hugging Face checkpoint (needs torch; no long-form check)")
    eval_p.add_argument("--dataset", required=True, help="Dataset directory made by `dataset build`")
    eval_p.add_argument("--split", default="test")
    eval_p.add_argument("--baseline", default="large-v3-turbo",
                        help="Zero-shot model scored on the same split through CT2: an alias or a CT2 directory")
    eval_p.add_argument("--language", default="fi")
    eval_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    eval_p.add_argument("--compute-type", default=None, help="CTranslate2 compute type (default: float16 on cuda, int8 on cpu)")
    eval_p.add_argument("--longform-audio", default=None, help="Long-form audio (e.g. a sermon) for the regression check")
    eval_p.add_argument("--longform-reference", default=None, help="Hand-corrected UTF-8 text of --longform-audio")
    eval_p.add_argument("--longform-tolerance", type=float, default=0.02,
                        help="Largest tolerated absolute WER increase over the baseline on the long-form audio")
    eval_p.add_argument("--min-improvement", type=float, default=0.0,
                        help="Candidate WER must be below baseline * (1 - this); the default only asks for better")
    eval_p.add_argument("--require-longform", action="store_true",
                        help="Fail the gate when no long-form audio and reference are given, instead of skipping that check")
    eval_p.add_argument("--out", default=None, help="Where to write the gate (default: gate.json in the --model directory)")
    eval_p.add_argument("--dump-predictions", default=None, metavar="PATH",
                        help="Write every reference and hypothesis as JSON lines. They are people's words: "
                             "off by default, and never part of gate.json")
    eval_p.set_defaults(func=_eval)

    models_p = sub.add_parser("models", help="Model registry: register exports, promote them, list what the service can serve")
    models_sub = models_p.add_subparsers(dest="models_command", required=True)
    registry_help = "Registry directory (default: env AUDITOR_STT_REGISTRY_DIR or ./models/registry)"

    reg_p = models_sub.add_parser("register", help="Add a CTranslate2 export to the registry as <name>@<version>")
    reg_p.add_argument("--name", required=True)
    reg_p.add_argument("--version", required=True)
    reg_p.add_argument("--export", required=True, help="CTranslate2 export directory (`export --out`); _merged_hf is never copied")
    reg_p.add_argument("--training-run", default=None, help="Training output directory; its training_metadata.json is recorded")
    reg_p.add_argument("--gate", default=None, help="gate.json from `eval` (default: gate.json inside --export)")
    reg_p.add_argument("--quantization", default=None, help="Quantization of the export, recorded in the metadata")
    reg_p.add_argument("--move", action="store_true", help="Remove the copied files from --export once registered")
    reg_p.add_argument("--registry-dir", default=None, help=registry_help)
    reg_p.set_defaults(func=_models_register)

    promote_p = models_sub.add_parser(
        "promote",
        help="Make a version the one `registry:<name>` serves (exit code 0 done, 3 refused by the gate, 1 error)",
    )
    promote_p.add_argument("--name", required=True)
    promote_p.add_argument("--version", required=True)
    promote_p.add_argument("--force", action="store_true", help="Promote even without a passed gate (recorded as forced)")
    promote_p.add_argument("--registry-dir", default=None, help=registry_help)
    promote_p.set_defaults(func=_models_promote)

    list_p = models_sub.add_parser("list", help="List registered models, their current and all versions")
    list_p.add_argument("--registry-dir", default=None, help=registry_help)
    list_p.set_defaults(func=_models_list)

    show_p = models_sub.add_parser("show", help="Show one model: current pointer, versions, and a version's metadata and gate")
    show_p.add_argument("--name", required=True)
    show_p.add_argument("--version", default=None, help="Default: the current version")
    show_p.add_argument("--registry-dir", default=None, help=registry_help)
    show_p.set_defaults(func=_models_show)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
