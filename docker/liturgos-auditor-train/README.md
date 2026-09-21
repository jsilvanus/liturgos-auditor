# Training Image for Liturgos Auditor STT

## Why a Separate Training Image?

PyTorch (torch) is excluded from the serving image to keep it lean and to work around a Windows-specific constraint: on the maintainer's machine, Smart App Control blocks PyTorch's native DLLs, making training impossible locally. This training image bundles PyTorch and runs in a Linux container instead, letting fine-tuning and model export happen without modification to the serving environment.

The pipeline itself (stages, behaviours, hardware, troubleshooting) is described in [docs/training-pipeline.md](../../docs/training-pipeline.md). What the data is, where it ends up and how to erase it is in [docs/data-protection.md](../../docs/data-protection.md); read it before syncing real recordings.

## Build

From the repo root, with build context `python-packages/liturgos-auditor-stt`:

```bash
# CPU variant (suitable for most development; no NVIDIA toolkit needed)
docker build -t auditor-train:cpu -f docker/liturgos-auditor-train/Dockerfile python-packages/liturgos-auditor-stt

# GPU variant (CUDA 12.8; requires NVIDIA Container Toolkit)
docker build -t auditor-train:cuda -f docker/liturgos-auditor-train/Dockerfile.cuda python-packages/liturgos-auditor-stt
```

The images contain the code and its dependencies only: no recordings, no model weights. There is no git checkout in the image, so `training_metadata.json` records `"git_commit": null`; tag the image yourself and note the tag with each run.

## Run

Mount three volumes:
- `<host data dir>:/data` — the ledger (`ledger.sqlite`), the normalised audio store (`audio/`) and the built datasets (`datasets/<version>/`). Read/write for every command that touches it: `dataset sync`, `dataset build`, `dataset lineage`, `dataset purge`, `dataset prune` (the ledger is opened read/write), and `train` writes cache files into the dataset it reads.
- `<host models dir>:/models` — training runs, exports and, if you keep it here, the registry (written by `train`, `export`, `eval` (its `gate.json`) and `models register`/`promote`).
- `auditor-hf:/hf` — HuggingFace Hub cache (downloaded base models); persist across runs

Optionally pass environment variables:
- `CSV_ADMIN_TOKEN` — for `dataset sync`. It is only the default name of the variable that holds the bearer token (`--token-env` changes it); the value can be csv's read-only export token or an admin JWT. Pass it with `-e CSV_ADMIN_TOKEN` (the value comes from the host environment) rather than writing it on the command line
- `AUDITOR_STT_SPEAKER_SALT` — optional, for `dataset sync`. Only used when an export carries a raw numeric user id (not the case with the current csv export, which sends a pseudonymous `speaker_id`); without it such ids are dropped and the recording is stored without a speaker
- `HF_HUB_OFFLINE=1` — after first run with internet access, set this to prevent re-downloads. The first `eval` also downloads its baseline model (`large-v3-turbo` by default) into `/hf`

The image sets `HF_HOME=/hf` and `AUDITOR_STT_TRAIN_DATA_DIR=/data`, so `--data-dir` can be left out.

`dataset build` puts its output in `/data/datasets/<version>` by default. Keep it there: `dataset lineage`, `dataset purge` and `dataset prune` look only in `<data dir>/datasets`, so a dataset built with `--out` somewhere else is invisible to the erasure and retention commands. `<version>` is a 16-character hash printed by `dataset build`.

### PowerShell examples

```powershell
# Sync a crowd-source-voice corpus into the local ledger
$env:CSV_ADMIN_TOKEN = "your-token-here"
docker run --rm -v $env:USERPROFILE\auditor-data:/data -v $env:USERPROFILE\auditor-models:/models -v auditor-hf:/hf `
  -e CSV_ADMIN_TOKEN `
  auditor-train:cpu dataset sync --base-url https://csv.example.org --corpus-id 1

# Build a versioned HF dataset from the synced ledger (prints the <version>)
docker run --rm -v $env:USERPROFILE\auditor-data:/data -v $env:USERPROFILE\auditor-models:/models -v auditor-hf:/hf `
  auditor-train:cpu dataset build --corpus-id 1

# Fine-tune on the built dataset with LoRA
docker run --rm -v $env:USERPROFILE\auditor-data:/data -v $env:USERPROFILE\auditor-models:/models -v auditor-hf:/hf `
  auditor-train:cpu train --dataset /data/datasets/<version> --out /models/run-001 --peft lora --gradient-checkpointing

# Export the run (merge LoRA if used)
docker run --rm -v $env:USERPROFILE\auditor-models:/models -v auditor-hf:/hf `
  auditor-train:cpu export --model /models/run-001 --out /models/exported-001 --merge-lora

# Evaluate: candidate against the zero-shot baseline on the test split; writes /models/exported-001/gate.json
# The optional long-form check needs a private audio file and its hand-corrected text (kept out of git, mounted read-only)
docker run --rm -v $env:USERPROFILE\auditor-data:/data -v $env:USERPROFILE\auditor-models:/models -v auditor-hf:/hf `
  -v $env:USERPROFILE\private-eval:/private:ro `
  auditor-train:cpu eval --model /models/exported-001 --dataset /data/datasets/<version> `
  --longform-audio /private/sermon.wav --longform-reference /private/sermon.txt

# Register the export and promote it (default registry: /models/registry, i.e. <host models dir>\registry)
docker run --rm -v $env:USERPROFILE\auditor-models:/models `
  auditor-train:cpu models register --name whisper-fi --version run-001 --export /models/exported-001 --training-run /models/run-001 --registry-dir /models/registry
docker run --rm -v $env:USERPROFILE\auditor-models:/models `
  auditor-train:cpu models promote --name whisper-fi --version run-001 --registry-dir /models/registry
docker run --rm -v $env:USERPROFILE\auditor-models:/models `
  auditor-train:cpu models list --registry-dir /models/registry
```

### Bash examples (Linux/macOS)

```bash
# Sync
export CSV_ADMIN_TOKEN="your-token-here"
docker run --rm -v ~/auditor-data:/data -v ~/auditor-models:/models -v auditor-hf:/hf \
  -e CSV_ADMIN_TOKEN \
  auditor-train:cpu dataset sync --base-url https://csv.example.org --corpus-id 1

# Build (prints the <version>)
docker run --rm -v ~/auditor-data:/data -v ~/auditor-models:/models -v auditor-hf:/hf \
  auditor-train:cpu dataset build --corpus-id 1

# Train
docker run --rm -v ~/auditor-data:/data -v ~/auditor-models:/models -v auditor-hf:/hf \
  auditor-train:cpu train --dataset /data/datasets/<version> --out /models/run-001 --peft lora --gradient-checkpointing

# Export
docker run --rm -v ~/auditor-models:/models -v auditor-hf:/hf \
  auditor-train:cpu export --model /models/run-001 --out /models/exported-001 --merge-lora

# Evaluate (exit code 0: gate passed, 3: gate failed, 1: error); gate.json is written into the export directory
docker run --rm -v ~/auditor-data:/data -v ~/auditor-models:/models -v auditor-hf:/hf \
  -v ~/private-eval:/private:ro \
  auditor-train:cpu eval --model /models/exported-001 --dataset /data/datasets/<version> \
  --longform-audio /private/sermon.wav --longform-reference /private/sermon.txt

# Register, promote, list, show
docker run --rm -v ~/auditor-models:/models \
  auditor-train:cpu models register --name whisper-fi --version run-001 \
  --export /models/exported-001 --training-run /models/run-001 --registry-dir /models/registry
docker run --rm -v ~/auditor-models:/models \
  auditor-train:cpu models promote --name whisper-fi --version run-001 --registry-dir /models/registry
docker run --rm -v ~/auditor-models:/models \
  auditor-train:cpu models list --registry-dir /models/registry
docker run --rm -v ~/auditor-models:/models \
  auditor-train:cpu models show --name whisper-fi --registry-dir /models/registry

# GPU variant (add --gpus all)
docker run --rm --gpus all -v ~/auditor-data:/data -v ~/auditor-models:/models -v auditor-hf:/hf \
  auditor-train:cuda train --dataset /data/datasets/<version> --out /models/run-001 --peft lora --gradient-checkpointing
```

Notes on `eval` and `models`:

- `eval` without `--longform-audio` and `--longform-reference` skips the long-form regression check, and a skipped check does not fail the gate. Add `--require-longform` to make the check mandatory. The check needs a hand-corrected reference that is not in git; see docs/training-pipeline.md, section 5.
- `models promote` refuses a version whose gate is missing or failed unless `--force` (recorded as forced).
- The service reads the registry from its own volume (`/models/registry` on the `auditor-stt-models` volume in `docker-compose.yml`). To register straight into it, mount that volume too and point `--registry-dir` at it, for example `-v liturgos-auditor_auditor-stt-models:/service-models` with `--registry-dir /service-models/registry` (Docker prefixes the compose project name; check the real name with `docker volume ls`). Only the CT2 export is copied there; training data, runs and `_merged_hf` stay in the training volumes. Then switch the running service with `POST /model` (needs `AUDITOR_STT_API_KEY`) or start it with `AUDITOR_STT_MODEL=registry:whisper-fi`.
- GPU evaluation: `eval` scores through CTranslate2, which in the CUDA image may need the CUDA 12 libraries; see the comments in `Dockerfile.cuda`. `--device cpu` works in either image.

### Erasure and retention commands

```bash
# Which dataset versions and models used a speaker (pseudonymous speaker id); --json for machine-readable output
docker run --rm -v ~/auditor-data:/data -v ~/auditor-models:/models \
  auditor-train:cpu dataset lineage --speaker <speaker id> --models-dir /models

# Erase the speaker from the ledger, audio store and dataset versions; dry run unless --yes. Models are only listed.
docker run --rm -v ~/auditor-data:/data -v ~/auditor-models:/models \
  auditor-train:cpu dataset purge --speaker <speaker id> --models-dir /models
docker run --rm -v ~/auditor-data:/data -v ~/auditor-models:/models \
  auditor-train:cpu dataset purge --speaker <speaker id> --models-dir /models --yes

# Retention: keep the newest N dataset versions and delete checkpoint-N directories of finished runs; dry run unless --yes
docker run --rm -v ~/auditor-data:/data -v ~/auditor-models:/models \
  auditor-train:cpu dataset prune --keep-datasets 2 --models-dir /models --runs-dir /models
```

The csv deletion and a `dataset sync` come first, or the next sync restores the recordings; the full procedure is in docs/data-protection.md, section 6. `--models-dir` must be given in the container: its default (`./models`) is not `/models` there.

## Privacy & Storage Notes

- **Data dir** holds voice recordings (`audio/`), the ledger (`ledger.sqlite`: pseudonymous speaker ids, prompt texts, hashes) and the built datasets (`datasets/`, which embed a copy of the audio). Mount only what a command needs; a command that does not use recordings (`export`, `models`) needs no data mount.
- **Models dir** holds training runs (final model or adapter and intermediate checkpoints), exports and, if you keep it here, the registry. Checkpoints and models can memorise training utterances: treat them with the same care as the data, and remove old checkpoints with `dataset prune --runs-dir` (see docs/data-protection.md).
- **After export**: for a LoRA run, `--merge-lora` leaves the merged full-precision checkpoint in `_merged_hf` inside the export directory. It is fp32 and large (about 3 GB for `large-v3-turbo`) and is not needed for serving. It can be deleted once you no longer need it (for example for `eval --backend hf`); `models register` never copies it, and only the CTranslate2 files in the export directory are used for serving.
- **Never push** datasets or fine-tuned models to a public hub without careful review of the training data provenance. The code has no push feature.
- **Local/on-premise only**: This image is designed for self-hosted training on the maintainer's infrastructure and is not suitable for cloud platforms with shared GPUs or untrusted network access.
- Encrypt the disk that holds the data and models directories, and keep them outside the git working tree (the repository `.gitignore` ignores only `/data/` and `/models/` at the repository root). Do not `docker commit` a container that has run with data.
- Keep the training volumes separate from the service's `auditor-stt-data` and `auditor-stt-models` volumes.
