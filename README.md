# liturgos-auditor

Self-hosted speech-to-text service (default language Finnish) for transcribing live streams and recorded services, plus the tooling to fine-tune the model on crowd-sourced recordings.

`liturgos-auditor` is a FastAPI + faster-whisper service with two kinds of callers:

- **lcyt** (live-captions-yt): live stream captions from short audio chunks posted to `/inference` (whisper.cpp-compatible) or `/v1/audio/transcriptions` (OpenAI-compatible).
- **saarnavideo**: full-length recordings (sermons, services) through the batch jobs API (`/v1/jobs`), which the code describes as written for saarnavideo. Finished chunks are kept on disk and a partial result can be fetched while a job runs.

Both share one loaded model. Live requests are served before the next batch chunk starts; a chunk that is already running is never interrupted.

## Components

- **Service** (`auditor_stt/serve/`): FastAPI + faster-whisper inference, two-priority queue, batch jobs (enabled by a data directory), model registry lookup and hot model switching.
- **Live inference** (`/inference`, `/v1/audio/transcriptions`): whisper.cpp-compatible and OpenAI-compatible endpoints.
- **Batch jobs** (`/v1/jobs`): submit a video/audio file (upload or a path under the server's media root), poll progress, fetch the result as JSON, VTT, SRT, plain text or YouTube Live Captions records.
- **Model registry** (`auditor-stt models ...`, `GET/POST /model`): fine-tuned models are registered and promoted with the CLI; the service serves them via `AUDITOR_STT_MODEL=registry:<name>[@<version>]` or `POST /model`.
- **Training pipeline** (`auditor-stt dataset|train|export|eval|models`): sync crowd-source-voice recordings into a local ledger, build versioned speaker-disjoint datasets, fine-tune (full or LoRA), export to CTranslate2, run the evaluation gate, register and promote models. Includes data protection tooling (`dataset lineage`, `dataset purge`, `dataset prune`). See [docs/training-pipeline.md](docs/training-pipeline.md).

## Quick start

### Development

```bash
cd python-packages/liturgos-auditor-stt
python3 -m venv .venv
. .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
```

Run the service:

```bash
auditor-stt serve --port 8090
```

`auditor-stt serve` listens on `0.0.0.0` and, unless `AUDITOR_STT_DATA_DIR` is already set, sets it to `./data` so batch jobs work out of the box. Starting the ASGI app any other way (for example `uvicorn auditor_stt.serve.app:app`) leaves jobs disabled unless you set `AUDITOR_STT_DATA_DIR` yourself; `/v1/jobs` then answers 503.

The model is downloaded on first start; until it is loaded `/health` answers 503.

Test it:

```bash
curl http://localhost:8090/health
```

### Docker Compose

Start the service with default settings (builds the CPU image; `AUDITOR_STT_DEVICE` defaults to `cpu` in the compose file):

```bash
docker compose up
```

The service listens on `http://localhost:8090` (or the port specified by `AUDITOR_STT_HOST_PORT`):

```bash
AUDITOR_STT_HOST_PORT=8097 docker compose up
```

To use a GPU (requires NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Test:

```bash
curl http://localhost:8090/health
```

Transcribe a WAV file:

```bash
curl -X POST http://localhost:8090/inference \
  -F "file=@sample.wav" \
  -F "language=fi"
```

Batch jobs are already enabled in the compose file (`AUDITOR_STT_DATA_DIR: /data`), so you can submit an upload:

```bash
curl -X POST http://localhost:8090/v1/jobs \
  -F "file=@sermon.mp4" \
  -F "language=fi"
```

To submit a file by path instead of uploading it, uncomment the media root lines in `docker-compose.yml` (and set `AUDITOR_STT_MEDIA_HOST_DIR` to the host directory):

```yaml
# AUDITOR_STT_MEDIA_ROOT: /media
# - ${AUDITOR_STT_MEDIA_HOST_DIR}:/media:ro
```

Then:

```bash
curl -X POST http://localhost:8090/v1/jobs \
  -F "source_path=/media/sermon.mp4" \
  -F "language=fi"
```

## All endpoints

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| `GET` | `/health` | Model status; 200 once loaded, 503 while loading or after a failed load | none |
| `GET` | `/status` | Live/batch queue depth and model info | if key set |
| `GET` | `/model` | Loaded model and the registry contents (models, current version, versions) | if key set |
| `POST` | `/model` | Switch model (JSON body `{"model": "<spec>"}`) | key required |
| `POST` | `/inference` | whisper.cpp-compatible: transcribe an uploaded audio segment | if key set |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible: transcribe an uploaded audio segment | if key set |
| `POST` | `/v1/jobs` | Submit a job (upload or media path); 202 | if key set |
| `GET` | `/v1/jobs` | List jobs, newest first (filter: `status`, `client_ref`) | if key set |
| `GET` | `/v1/jobs/{id}` | Job status and progress | if key set |
| `GET` | `/v1/jobs/{id}/result` | Transcript as `json`, `vtt`, `srt`, `text` or `youtube` | if key set |
| `DELETE` | `/v1/jobs/{id}` | Cancel a running job (202) or delete a job (204) | if key set |

"If key set" means: when `AUDITOR_STT_API_KEY` is set the request needs `Authorization: Bearer <key>` (otherwise 401); when it is not set the endpoint is open, `/inference` and `/v1/audio/transcriptions` included. Only `/health` never needs a key. `POST /model` is different: it answers 403 when no key is configured, and it only accepts `registry:<name>[@<version>]` specs and aliases listed in `AUDITOR_STT_ALLOWED_MODELS`.

Live routes (`/inference`, `/v1/audio/transcriptions`) take a multipart `file` and accept `response_format` = `json` (default), `verbose_json` or `text`; any other value (including `vtt`/`srt`) is a 400. `/inference` returns the full result (`text`, `language`, `segments[]` with words) for both `json` and `verbose_json`; `/v1/audio/transcriptions` returns only `{"text": ...}` for `json` and the full result for `verbose_json`. Only `/inference` accepts `vad`. Caption formats (VTT, SRT, YouTube) exist only on the jobs result route. Details: [word-level transcription](docs/word-level-transcription.md), [integration guide](docs/integration.md), [batch jobs API](docs/batch-jobs.md).

## Environment variables

### Service

| Variable | Default | Notes |
|----------|---------|-------|
| `AUDITOR_STT_MODEL` | `large-v3-turbo` | faster-whisper model alias, path, or `registry:<name>[@<version>]` |
| `AUDITOR_STT_MODEL_DIR` | (none) | Download/cache directory for models; when unset faster-whisper uses its own default cache. Use a persistent volume |
| `AUDITOR_STT_DEVICE` | `auto` | `cpu`, `cuda` or `auto`. `auto` tries CUDA (float16) first and falls back to CPU (int8) on any failure including missing CUDA runtime libraries; `cuda` and `cpu` do not fall back |
| `AUDITOR_STT_COMPUTE_TYPE` | (none) | faster-whisper compute type (`float16`, `int8`, `int8_float16`, ...). When unset: `float16` on CUDA, `int8` on CPU |
| `AUDITOR_STT_DEFAULT_LANGUAGE` | `fi` | Language used when a request does not send `language` |
| `AUDITOR_STT_PORT` | `8090` | Listen port. Read only by `auditor-stt serve` (default of `--port`) |
| `AUDITOR_STT_API_KEY` | (none) | Bearer token. When set, every endpoint except `/health` requires it; when unset all endpoints are open. Required for `POST /model` |
| `AUDITOR_STT_MAX_QUEUE` | `8` | Live requests allowed in the queue (the running one plus waiting ones) before a new live request is rejected with 503. Batch chunks are not counted and never rejected |
| `AUDITOR_STT_DATA_DIR` | (none) | Job storage (`<dir>/jobs`). Unset: `/v1/jobs` answers 503. `auditor-stt serve` defaults it to `./data`; the Docker images set `/data` |
| `AUDITOR_STT_MEDIA_ROOT` | (none) | Directory that `source_path` submissions must point into. Unset: `source_path` is rejected with 422 |
| `AUDITOR_STT_SOURCE_URL_HOSTS` | (none) | Comma-separated host names (`*` patterns allowed) that `source_url` submissions may be fetched from. Unset: `source_url` is rejected with 422 |
| `AUDITOR_STT_SOURCE_URL_TIMEOUT` | `300` | Seconds the service waits on the host when fetching a `source_url` |
| `AUDITOR_STT_STRIP` | `local` | `fleet` makes `source_url` jobs strip their audio on an fffleet worker instead of downloading the file here ([details](docs/batch-jobs.md#stripping-the-audio-on-an-fffleet-worker)). Needs `AUDITOR_STT_FLEET_URL` and `AUDITOR_STT_PUBLIC_URL` |
| `AUDITOR_STT_FLEET_URL` | (none) | Base URL of the fffleet orchestrator (or a single worker) |
| `AUDITOR_STT_FLEET_TOKEN` / `AUDITOR_STT_FLEET_CLIENT_ID` + `AUDITOR_STT_FLEET_CLIENT_SECRET` | (none) | A static bearer token, or OAuth2 client credentials (scope `jobs`) for the fleet |
| `AUDITOR_STT_PUBLIC_URL` | (none) | How fleet workers reach this service; the stripped audio is PUT to `<url>/v1/jobs/{id}/audio` |
| `AUDITOR_STT_FLEET_FALLBACK` | `on` | `off`: fail the job instead of stripping locally when the fleet is unreachable |
| `AUDITOR_STT_FLEET_POLL_SECONDS` / `AUDITOR_STT_FLEET_TIMEOUT_SECONDS` | `2` / `21600` | Fleet job polling interval and the longest a strip may take |
| `AUDITOR_STT_LIVE_SOURCE_HOSTS` | (none) | Comma-separated host names (`*` patterns allowed) that live sessions may pull from. Unset: `/v1/live` is off ([details](docs/live-sessions.md)) |
| `AUDITOR_STT_LIVE_REQUIRES` | (none) | fffleet capabilities a live stream job needs, e.g. `net:mediamtx` |
| `AUDITOR_STT_MAX_LIVE_SESSIONS` | `2` | Concurrent live sessions (one model is shared) |
| `AUDITOR_STT_LIVE_RECONNECT_SECONDS` | `120` | How long a session keeps retrying a lost stream before it ends |
| `AUDITOR_STT_LIVE_MAX_SEGMENT_SECONDS` / `_PAUSE_SECONDS` | `10` / `0.6` | Longest segment, and the pause that ends one |
| `AUDITOR_STT_LIVE_MAX_LAG_SECONDS` | `30` | A segment still waiting for the model after this long is dropped (status event) |
| `AUDITOR_STT_LIVE_CLOCK_DRIFT_SECONDS` | `1` | Drift between stream position and wall time that moves the clock anchor |
| `AUDITOR_STT_LIVE_READ_TIMEOUT` | `120` | Seconds without PCM from the fleet before the stream counts as lost |
| `AUDITOR_STT_MAX_INGEST_MB` | `4096` | Size cap for the WAV a fleet worker sends back |
| `AUDITOR_STT_MAX_UPLOAD_MB` | `2048` | Size cap (1 MB = 1024 x 1024 bytes) for uploads to `POST /v1/jobs` |
| `AUDITOR_STT_MAX_LIVE_UPLOAD_MB` | `64` | Size cap for uploads to `POST /inference` and `POST /v1/audio/transcriptions` (live routes) |
| `AUDITOR_STT_JOB_TTL_HOURS` | `72` | How long a finished job (completed, failed or cancelled) and its result are kept before they are purged (checked at startup and hourly) |
| `AUDITOR_STT_CARRY_CONTEXT_CHARS` | `200` | Batch jobs: how many characters of the previous chunk's text are passed as prompt context to the next chunk; `0` turns it off |
| `AUDITOR_STT_BATCH_VAD` | `true` | Batch jobs: VAD filter inside each chunk. `0`, `false`, `no`, `off` or empty disable it |
| `AUDITOR_STT_REGISTRY_DIR` | `./models/registry` | Directory holding registered model versions and their `current.json` pointers; read by the service and by `auditor-stt models` |
| `AUDITOR_STT_ALLOWED_MODELS` | (none) | Comma-separated entries (aliases) accepted by `POST /model` in addition to `registry:` specs |

### Training side

Read by the `auditor-stt dataset` commands, not by the service: `AUDITOR_STT_TRAIN_DATA_DIR` (ledger and dataset directory, default `./data`), `AUDITOR_STT_SPEAKER_SALT` (salt for pseudonymising speaker ids, `dataset sync`) and `CSV_ADMIN_TOKEN` (crowd-source-voice bearer token, `dataset pull` and `dataset sync`). The salt and token variable names are the defaults of `--speaker-salt-env` and `--token-env`. `auditor-stt eval` also reads `AUDITOR_STT_MODEL_DIR`. See [docs/training-pipeline.md](docs/training-pipeline.md).

The `scripts/video-to-vtt-jobs.py` client (jobs API) has its own variables (`AUDITOR_STT_URL`, and the API key variable named by `--api-key-env`); see [docs/video-to-vtt-jobs.md](docs/video-to-vtt-jobs.md).

## Documentation

- **[Integration guide](docs/integration.md)**: How lcyt and saarnavideo call the service.
- **[Batch jobs API](docs/batch-jobs.md)**: Reference for `/v1/jobs` submission, polling, results, retention and error codes.
- **[Word-level transcription](docs/word-level-transcription.md)**: Segment and word fields, `avg_logprob`, live route options.
- **[video-to-vtt script](docs/video-to-vtt.md)**: Client-side chunking: cuts the video into overlapping chunks, posts each to `/inference` and appends VTT cues as it goes (`scripts/video-to-vtt.py`; works with any service version).
- **[video-to-vtt-jobs script](docs/video-to-vtt-jobs.md)**: Client for the server-side jobs API with resume and partial results (`scripts/video-to-vtt-jobs.py`).
- **[Model download](docs/model-download.md)**: Fetching faster-whisper models ahead of time.
- **[Docker README](docker/liturgos-auditor-stt/README.md)**: Serving image build, environment, volumes, GPU.
- **[Training pipeline](docs/training-pipeline.md)**: Dataset sync and build, fine-tuning, export, evaluation gate, registry and promotion.
- **[Data protection](docs/data-protection.md)**: Personal data handling in the training pipeline (ledger, speaker pseudonyms, purge, retention).
- **[Legal items still open](LEGAL-TODO.md)**: What must be settled with the owner and a DPO before any real recordings are synced.
- **[crowd-source-voice contract](docs/crowd-source-voice-contract.md)**: What the pipeline expects from the crowd-source-voice export API.
- **[Training image README](docker/liturgos-auditor-train/README.md)**: Docker image for the training commands.
- **[auditor-stt package README](python-packages/liturgos-auditor-stt/README.md)**: Installation extras and CLI overview.

## Development

Tests require Python 3.10+ (`requires-python` in `pyproject.toml`). The CPU image is built on `python:3.12-slim`; the CUDA image uses the `python3` of its Ubuntu base image.

```bash
cd python-packages/liturgos-auditor-stt
pip install -e ".[dev]"
pytest -v
```

**Windows: path length**: Downloading a model into a deeply nested `AUDITOR_STT_MODEL_DIR` can fail on Windows because of the 260-character path limit. Hugging Face blob names are long, and deeply nested directories can cause paths to exceed this limit. Use a short directory such as `C:\models` or enable long paths via the Windows registry (see Microsoft documentation). If model download fails with a path error, check `/health` in the logs for the full error.

The slow end-to-end training smoke test (needs the `training` extra and downloads `openai/whisper-tiny`) is skipped unless enabled:

```bash
AUDITOR_STT_RUN_SLOW=1 pytest -v tests/test_training_smoke.py
```

## Architecture notes

- **One model at a time**: the service serves a single loaded model. `POST /model` loads the new model beside the running one and swaps it in only when it has loaded; a failed load leaves the current model serving, and requests already running finish on the model they started with. Both models are in memory during the switch. A batch job picks up the new model at its next chunk, so one job's chunks can come from different models (the result lists them in `models`).
- **Two-priority queue**: one worker runs inference calls one at a time. Live requests are served before the next batch chunk starts; a chunk that is already running is never interrupted, so a live request can wait for the batch chunk in progress. On CPU-only deployments use a smaller `chunk_seconds` for batch jobs or run batch jobs on a separate instance. For scale: the lcyt project's plan (`docs/plans/plan_local_stt.md` in live-captions-yt) estimates a real-time factor of roughly 0.2-0.4 for `large-v3-turbo` on CPU with int8 and 8 modern vCPUs. That is an estimate from that plan, not something measured in this repository.
- **Checkpointed jobs**: every finished chunk is written to disk atomically, and the chunk plan is stored and never recomputed. After a crash or restart the service requeues jobs that were running and continues from the chunk files; at most the chunk in flight is redone. Jobs run one at a time in submission order. Each chunk is attempted up to three times before the job is marked failed.
- **Job data**: a job's uploaded copy and its normalised audio are deleted as soon as the job ends (completed, failed or cancelled). The transcript chunks and manifest stay until `AUDITOR_STT_JOB_TTL_HOURS` after the end or until `DELETE`. A file submitted by `source_path` is never modified or deleted. Nothing in the service or the training code reads job data; the training pipeline takes its data from crowd-source-voice.

## License

The code in this repository is MIT (see [LICENSE](LICENSE) and `python-packages/liturgos-auditor-stt/pyproject.toml`).

**The MIT license of the code does not extend to the models.** The repository contains no model weights; they are downloaded from Hugging Face at run time or produced by the training pipeline, and each carries its own license:

- **Default model** (`large-v3-turbo`): faster-whisper resolves the alias to the CTranslate2 conversion `mobiuslabsgmbh/faster-whisper-large-v3-turbo` of OpenAI's `openai/whisper-large-v3-turbo`. OpenAI publishes Whisper code and weights under MIT, and the conversions of the models named in this repository (`Systran/faster-whisper-*`, `mobiuslabsgmbh/faster-whisper-large-v3-turbo`) are published under MIT as well. The license of a downloaded model is whatever its Hugging Face model card says, so check the card of the exact model you deploy.
- **Training base models** (`--preset`, `auditor-stt train`): `openai/whisper-large-v3-turbo` and `openai/whisper-small` (MIT). You can pass any other `--model`; its license then applies.
- **Fine-tuned models** (`train`, `export`, `models register`) are derived from a base model and from your training data. They inherit the base model's license terms, and in addition **the terms under which the recordings were collected may restrict how the resulting model can be used, shared or published**. Nothing in this repository grants those rights. For models trained on crowd-source-voice recordings, the consent text, privacy policy and data license of that corpus decide (see [LEGAL-TODO.md](LEGAL-TODO.md)); do not publish or redistribute such a model before they are settled. A fine-tune of a model from another source (for example a community fine-tune with a non-commercial license) can be more restrictive than MIT.
- **Dependencies** (faster-whisper, CTranslate2, Transformers, PEFT, PyTorch and others) are separate packages under their own licenses; see each project.

See [NOTICE](NOTICE) for the list. This is a maintainer's summary, not legal advice.
