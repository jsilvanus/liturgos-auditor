# auditor-stt Docker images

Two serving images are provided: CPU-only (`Dockerfile`, based on `python:3.12-slim`) and GPU (`Dockerfile.cuda`, based on `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`). Both install ffmpeg and the `auditor-stt` package (serving dependencies only) and start `auditor-stt serve`. Model weights are not baked into either image; they are downloaded on first start. Fine-tuning has its own image: [docker/liturgos-auditor-train](../liturgos-auditor-train/README.md). Both images run as an unprivileged user, not as root; see [Non-root user](#non-root-user), and [Upgrading an existing deployment](#upgrading-an-existing-deployment) if you already have volumes from an older image.

## Building

From the repository root:

```bash
# CPU image
docker build -t auditor-stt:cpu -f docker/liturgos-auditor-stt/Dockerfile \
  python-packages/liturgos-auditor-stt

# GPU image (requires NVIDIA Container Toolkit)
docker build -t auditor-stt:gpu -f docker/liturgos-auditor-stt/Dockerfile.cuda \
  python-packages/liturgos-auditor-stt
```

Or use Docker Compose from the repository root, which builds and runs the **CPU** image (the GPU image is not wired into `docker-compose.yml`):

```bash
docker compose up
```

The build context is `python-packages/liturgos-auditor-stt`. Its `.dockerignore` is an allow-list: only `pyproject.toml`, `README.md` and `auditor_stt/` (without bytecode) can enter a build, so a local virtualenv, a `data/` or `models/` directory, a ledger or voice recordings in that directory never reach the daemon or an image. If a Dockerfile ever needs another file from there, add it to that `.dockerignore`. The repository root has a `.dockerignore` as well, a deny-list for builds that use the root as context.

## Volumes

| Mount point | Purpose | Notes |
|---|---|---|
| `/models` | Model cache | Persist downloaded models and fine-tuned versions across container restarts. Mounted as `auditor-stt-models` in `docker-compose.yml`. Model downloads only go there when `AUDITOR_STT_MODEL_DIR=/models` is set (compose sets it; the images do not) |
| `/data` | Job storage | Job manifests, chunk results and working files (`<data dir>/jobs`). Mounted as `auditor-stt-data` in `docker-compose.yml`. Both images already set `AUDITOR_STT_DATA_DIR=/data`, so jobs are enabled |
| `/media` | Shared media (optional) | Read-only media directory for `source_path` submissions. Mount it and set `AUDITOR_STT_MEDIA_ROOT=/media`. Example: `-v /srv/media:/media:ro` |

Example manual run with volumes:

```bash
docker run --rm \
  -p 8090:8090 \
  -v auditor-stt-models:/models \
  -e AUDITOR_STT_MODEL_DIR=/models \
  -v auditor-stt-data:/data \
  -e AUDITOR_STT_DATA_DIR=/data \
  -v /srv/media:/media:ro \
  -e AUDITOR_STT_MEDIA_ROOT=/media \
  auditor-stt:cpu
```

## Non-root user

Both images run as the unprivileged user `auditor`, uid and gid `10001`, never as root (`docker inspect --format "{{.Config.User}}" <image>` prints `10001:10001`; the id is numeric so that orchestrator checks such as Kubernetes `runAsNonRoot` can verify it). The process starts as that user; there is no entrypoint that starts as root and drops privileges. The code in `/app` is owned by root and cannot be modified by the service. The service writes only to `/data`, `/models` (including `/models/registry`), its home directory `/home/auditor` (library caches, `HOME`) and `/tmp`.

- **Named volumes** (what `docker-compose.yml` uses): a volume that is created empty copies the owner of its mount point from the image, so a new volume is writable without any setup.
- **Bind mounts** (`-v /srv/auditor/models:/models`) keep the owner of the host directory. On a Linux host that directory must be writable by uid 10001 (`sudo chown -R 10001:10001 /srv/auditor/models`), or you run the container as the directory's owner with `--user "$(id -u):$(id -g)"` (Compose: `user: "1000:1000"`). Docker Desktop on Windows and macOS does not enforce host permissions on bind mounts, so this only matters on Linux hosts. With `--user` the home directory is world-writable, so the caches work for any uid; the files the service creates then belong to that uid.
- `docker-compose.yml` also sets `no-new-privileges` and drops all Linux capabilities. It does not make the root filesystem read-only, because the service writes uploads and normalised audio to `/tmp`.

### Upgrading an existing deployment

Volumes that were created by an older image, which ran as root, stay owned by root, so the non-root container cannot write to them until their ownership is fixed once. What you see if you skip this:

- a root-owned `/models` volume: the container runs but `GET /health` stays `503` (`"loaded": false`) and the log shows `PermissionError: [Errno 13] Permission denied: '/models/models--Systran--faster-whisper-...'` followed by `Model failed to load at startup`;
- a root-owned `/data` volume: the container exits at startup with `PermissionError: [Errno 13] Permission denied: '/data/jobs'` (with `restart: unless-stopped` it keeps restarting).

The fix is one `chown` per volume, run as root in a throwaway container, with the service stopped. Example for the default Compose project `liturgos-auditor` (Compose prefixes volume names with the project name, the directory name by default, and names the image `<project>-<service>`; check with `docker volume ls` and `docker images`, and use `auditor-stt:cpu` or whatever tag you built yourself instead of the image name below):

```bash
docker compose build
docker compose stop auditor-stt

docker run --rm --user 0 --entrypoint chown \
  -v liturgos-auditor_auditor-stt-models:/models \
  liturgos-auditor-auditor-stt -R 10001:10001 /models
docker run --rm --user 0 --entrypoint chown \
  -v liturgos-auditor_auditor-stt-data:/data \
  liturgos-auditor-auditor-stt -R 10001:10001 /data

docker compose up -d
```

Use plain `docker run` for this, not `docker compose run`: the Compose service drops all capabilities, and root without `CAP_CHOWN` cannot change owners. A volume that does not exist yet (for example the data volume, if your old container had none) needs no fix: Compose creates it from the new image, owned by 10001. On Windows with Git Bash, set `MSYS_NO_PATHCONV=1` for the commands, or Git Bash rewrites `/models` into a Windows path. You can check the result with `docker run --rm --entrypoint ls -v liturgos-auditor_auditor-stt-models:/models liturgos-auditor-auditor-stt -ldn /models`, which should show `10001 10001`.

## Environment variables

Set them with `-e` on `docker run`. For `docker compose`, only the variables listed in the `environment` block of `docker-compose.yml` reach the container; a `.env` file or your shell can supply the `${...}` values of those, but a variable that the compose file does not list has to be added to the file first.

| Variable | Code default | Set by the images | In `docker-compose.yml` | Notes |
|----------|--------------|-------------------|-------------------------|-------|
| `AUDITOR_STT_MODEL` | `large-v3-turbo` | — | `${AUDITOR_STT_MODEL:-large-v3-turbo}` | faster-whisper model alias, path, or `registry:<name>[@<version>]` |
| `AUDITOR_STT_MODEL_DIR` | none (faster-whisper's default cache, lost with the container) | — | `/models` | Model download/cache directory; should be a persistent volume |
| `AUDITOR_STT_DEVICE` | `auto` | CPU image `cpu`; GPU image `auto` | `${AUDITOR_STT_DEVICE:-cpu}` | `cpu`, `cuda` or `auto` (see [GPU image](#gpu-image)) |
| `AUDITOR_STT_COMPUTE_TYPE` | `float16` on CUDA, `int8` on CPU | — | not passed | faster-whisper compute type |
| `AUDITOR_STT_DEFAULT_LANGUAGE` | `fi` | `fi` | `${AUDITOR_STT_DEFAULT_LANGUAGE:-fi}` | Language used when a request sends none |
| `AUDITOR_STT_PORT` | `8090` | `8090` | not passed | Listen port of `auditor-stt serve`; if you change it, change the port mapping too |
| `AUDITOR_STT_API_KEY` | none | — | `${AUDITOR_STT_API_KEY:-}` | Bearer token; empty or unset means no authentication |
| `AUDITOR_STT_MAX_QUEUE` | `8` | — | not passed | Live requests allowed in the queue before a new one is rejected with 503 |
| `AUDITOR_STT_DATA_DIR` | none (jobs off) | `/data` | `/data` | Job storage; when unset `/v1/jobs` answers 503 |
| `AUDITOR_STT_MAX_UPLOAD_MB` | `2048` | — | `${AUDITOR_STT_MAX_UPLOAD_MB:-2048}` | Size cap for uploads to `POST /v1/jobs` (1 MB = 1024 x 1024 bytes) |
| `AUDITOR_STT_MAX_LIVE_UPLOAD_MB` | `64` | — | not passed | Size cap for uploads to live routes: `POST /inference` and `POST /v1/audio/transcriptions` |
| `AUDITOR_STT_JOB_TTL_HOURS` | `72` | — | `${AUDITOR_STT_JOB_TTL_HOURS:-72}` | How long finished jobs and their results are kept |
| `AUDITOR_STT_MEDIA_ROOT` | none | — | commented out | Directory `source_path` submissions must be inside; unset means `source_path` is rejected with 422 |
| `AUDITOR_STT_CARRY_CONTEXT_CHARS` | `200` | — | not passed | Batch jobs: characters of the previous chunk's text carried into the next chunk's prompt (`0` turns it off) |
| `AUDITOR_STT_BATCH_VAD` | `true` | — | not passed | Batch jobs: VAD filter inside each chunk (`0`, `false`, `no`, `off`, empty disable it) |
| `AUDITOR_STT_REGISTRY_DIR` | `./models/registry` (relative to the working directory, `/app` in the images) | — | `/models/registry` | Registered fine-tuned model versions; compose keeps them in the same volume as the model cache |
| `AUDITOR_STT_ALLOWED_MODELS` | none | — | `${AUDITOR_STT_ALLOWED_MODELS:-}` | Comma-separated aliases accepted by `POST /model` besides `registry:` specs |

## API key

To require authentication on every endpoint except `/health`:

```bash
docker run -e AUDITOR_STT_API_KEY=your_secret ...
```

Without a key, all endpoints are open, including `/inference` and `/v1/audio/transcriptions`. The `/health` endpoint never needs a key.

Clients send the key as an `Authorization: Bearer` header:

```bash
curl -H "Authorization: Bearer your_secret" http://localhost:8090/v1/jobs
```

## GPU image

The GPU image (`Dockerfile.cuda`) requires the **NVIDIA Container Toolkit** on the host.

Build:

```bash
docker build -t auditor-stt:gpu -f docker/liturgos-auditor-stt/Dockerfile.cuda \
  python-packages/liturgos-auditor-stt
```

Run with `--gpus all`:

```bash
docker run --rm \
  --gpus all \
  -p 8090:8090 \
  -v auditor-stt-models:/models \
  -e AUDITOR_STT_MODEL_DIR=/models \
  -v auditor-stt-data:/data \
  -e AUDITOR_STT_DATA_DIR=/data \
  auditor-stt:gpu
```

The GPU image sets `AUDITOR_STT_DEVICE=auto`: it tries CUDA (float16) first and falls back to CPU (int8) if that fails, so it also starts on a host without a GPU. `AUDITOR_STT_DEVICE=cuda` (or `cpu`) does not fall back: if the model cannot be loaded, `/health` stays at 503. `GET /health` reports the `device` in use.

## Model switching

Load a model at startup via `AUDITOR_STT_MODEL`:

```bash
-e AUDITOR_STT_MODEL=base
-e AUDITOR_STT_MODEL=registry:finnish@v1.2
```

`registry:<name>` without a version serves the version that was promoted with `auditor-stt models promote`; `registry:<name>@<version>` names one. Registered versions are read from `AUDITOR_STT_REGISTRY_DIR`.

Switch at runtime with `POST /model`:

```bash
curl -X POST http://localhost:8090/model \
  -H "Authorization: Bearer your_secret" \
  -H "Content-Type: application/json" \
  -d '{"model": "registry:finnish@v1.2"}'
```

- `POST /model` only works when `AUDITOR_STT_API_KEY` is set; otherwise it answers 403.
- It accepts `registry:` specs and the aliases listed in `AUDITOR_STT_ALLOWED_MODELS` (never a path unless you list it there); anything else is a 422.
- The new model is loaded next to the running one and swapped in once loaded, so both are in memory during the switch. A failed load answers 500 and the current model keeps serving. A switch while another one is running answers 409.
- `GET /model` shows the loaded model and what the registry holds.

```bash
-e AUDITOR_STT_ALLOWED_MODELS=base,small,medium
```

## Logs

View service logs:

```bash
docker logs -f <container-id>
```

The service logs startup, model loading, model switches, and per-job messages (job id, chunk index, timing, errors). The service code does not log transcript text or audio.

## Health check

```bash
curl http://localhost:8090/health
```

- **200 OK**: the model is loaded. Body: `{"status": "ok", "model_id": ..., "device": ..., "compute_type": ..., "loaded": true}`.
- **503 Service Unavailable**: the model is loading or failed to load. Same body with `"status": "loading"` and `"loaded": false`.

`docker-compose.yml` defines a health check that uses Python's `urllib` to avoid requiring `curl`. The container initially reports `unhealthy` while the model is downloading and loading; the `start_period: 300s` allows sufficient time for the largest models (300 seconds is generous for most cases). Once the model is loaded, the service becomes `healthy`. You can check the status with:

```bash
docker inspect --format "{{.State.Health.Status}}" <container-id>
```

## Storage and cleanup

**Models**: downloaded models are cached in `/models` (when `AUDITOR_STT_MODEL_DIR=/models`) and the service never deletes them.

**Jobs** (`<data dir>/jobs`):
- When a job ends (completed, failed or cancelled), its uploaded copy and normalised audio are deleted at once.
- The manifest and chunk results (the transcript) are kept for `AUDITOR_STT_JOB_TTL_HOURS` (default 72) after the job ended, then the job is deleted. The purge runs at startup and then hourly.
- `DELETE /v1/jobs/{id}` removes a job completely.
- Files submitted with `source_path` are on the media volume and are never modified or deleted.

## Networking

`auditor-stt serve` listens on `0.0.0.0` inside the container, on port 8090 unless `AUDITOR_STT_PORT` is changed. Port mapping is required to reach it from the host or other containers.

For Compose:

```yaml
ports:
  - "8090:8090"  # host:container
```

For manual run:

```bash
docker run -p 8090:8090 ...
```

## Resource limits

The code and this repository state no memory or CPU requirements, and none were measured for this README. What is known from the code:

- During a `POST /model` switch the old and the new model are in memory at the same time (on GPU as well).
- A batch job reads its audio chunk by chunk from a WAV on disk, not the whole file into memory.
- A live request is read into memory whole; the upload cap (`AUDITOR_STT_MAX_UPLOAD_MB`) applies to `POST /v1/jobs` only.
- A live request can wait for the batch chunk that is running. On CPU-only hosts use a smaller `chunk_seconds` for jobs or a separate instance for batch work.

Set container limits with Docker's own options (`--memory`, `--cpus`) or in the compose file, sized for the model you run.
