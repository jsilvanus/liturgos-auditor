# auditor-stt

Liturgos Auditor self-hosted speech-to-text service (faster-whisper) and the pipeline to fine-tune its model.

## Installation

Install the service:

```bash
pip install -e .
```

The base install brings the serving dependencies (FastAPI, uvicorn, python-multipart, faster-whisper, numpy, httpx). Batch jobs also need `ffmpeg` on the server's `PATH` (the Docker images install it).

Optional extras (from `pyproject.toml`):

- `dev`: pytest, pytest-asyncio, pytest-cov and `datasets`, for running the tests
- `dataset`: `datasets`, for `dataset build`
- `training`: `datasets`, torch, transformers, accelerate, peft, jiwer, for model training

```bash
pip install -e ".[dev]"
pip install -e ".[dataset]"
pip install -e ".[training]"
```

## CLI

The command list below is taken from `auditor-stt --help` and each sub-command's `--help`; run those for every option.

### Service

```bash
auditor-stt serve [--port PORT]
```

Runs the FastAPI server with uvicorn on `0.0.0.0`. The port defaults to `AUDITOR_STT_PORT`, else 8090. If `AUDITOR_STT_DATA_DIR` is not set it is set to `./data`, which enables batch jobs. Service configuration is by environment variables; see the [repository README](../../README.md#environment-variables).

### Dataset pipeline (`auditor-stt dataset ...`)

- `dataset pull`: Pull a validated snapshot from crowd-source-voice into a directory (`--base-url`, `--corpus-id`, `--out` are required; the bearer token comes from the environment variable named by `--token-env`, default `CSV_ADMIN_TOKEN`).
- `dataset sync`: Sync a crowd-source-voice corpus into the local ledger (incremental). `--data-dir` defaults to `AUDITOR_STT_TRAIN_DATA_DIR`, else `./data`. Raw user ids are pseudonymised with the salt from the variable named by `--speaker-salt-env` (default `AUDITOR_STT_SPEAKER_SALT`); without a salt they are stored speaker-less. `--allow-mass-removal` permits tombstoning more than half of a corpus's recordings in one sync.
- `dataset build`: Build a versioned Hugging Face dataset with a stable speaker-disjoint split, from the ledger or, with `--snapshot` (and `--out`), from a `dataset pull` snapshot. `--seed` defaults to 42; `--force` rebuilds an existing version.
- `dataset lineage`: List the dataset versions and models that used a speaker (`--speaker` is a pseudonymous speaker id; `--json` for machine-readable output).
- `dataset purge`: Erase a speaker from the ledger, the audio cache and the dataset versions. Dry run unless `--yes`. Affected models are only listed, not changed. Delete upstream first, or the next sync restores the recordings.
- `dataset prune`: Retention: delete old dataset versions (`--keep-datasets N`) and intermediate training checkpoints (`--runs-dir DIR`). Dry run unless `--yes`; dataset versions that a model still names are kept unless `--include-referenced`.

### Training pipeline

- `train`: Fine-tune Whisper on a built dataset (`--dataset`, `--out`). `--preset` is `large-v3-turbo` (default) or `small`; `--peft lora` trains adapters instead of all weights.
- `export`: Convert a Transformers Whisper checkpoint to CTranslate2 (`--model`, `--out`; `--quantization` is `float16` (default), `int8_float16` or `int8`; `--merge-lora` is required for adapter directories).
- `eval`: Evaluation gate. Scores a candidate against the zero-shot baseline on the test split and writes `gate.json`. Exit code 0: gate passed, 3: gate failed, 1: error.

### Model registry (`auditor-stt models ...`)

- `models register`: Add a CTranslate2 export to the registry as `<name>@<version>` (`--name`, `--version`, `--export`; `--training-run`, `--gate`, `--quantization`, `--move` optional).
- `models promote`: Make a version the one that `registry:<name>` serves (`--name`, `--version`). Exit code 0: done, 3: refused by the gate (no readable or passed `gate.json`), 1: error. `--force` promotes anyway and records it as forced.
- `models list`: List registered models with their current and all versions.
- `models show`: Show one model: current pointer, versions, and a version's metadata and gate (`--name`; `--version` defaults to the current one).

All `models` commands take `--registry-dir` (default: `AUDITOR_STT_REGISTRY_DIR`, else `./models/registry`).

More: [training pipeline](../../docs/training-pipeline.md), [data protection](../../docs/data-protection.md), [crowd-source-voice contract](../../docs/crowd-source-voice-contract.md), [repository README](../../README.md).
