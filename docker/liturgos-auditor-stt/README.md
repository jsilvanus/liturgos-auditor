# auditor-stt

faster-whisper (CTranslate2) inference server for Liturgos Auditor's self-hosted STT.

Two variants from one source tree:

- **`Dockerfile`** — CPU only (`python:3.12-slim` + ffmpeg). Runs anywhere.
- **`Dockerfile.cuda`** — GPU (`nvidia/cuda` runtime base). Requires the NVIDIA Container Toolkit + GPU access.

Model weights are **not** baked into either image — they download into `AUDITOR_STT_MODEL_DIR` (mount a volume there) on first boot. The default model is faster-whisper's `large-v3-turbo` alias.

## Build

The build context is the package source (`python-packages/liturgos-auditor-stt`):

```bash
# from repo root
docker build -t auditor-stt:local -f docker/liturgos-auditor-stt/Dockerfile python-packages/liturgos-auditor-stt
docker build -t auditor-stt:cuda -f docker/liturgos-auditor-stt/Dockerfile.cuda python-packages/liturgos-auditor-stt
```

## Run

```bash
docker run --rm -p 8090:8090 -v auditor-stt-models:/models \
  -e AUDITOR_STT_MODEL_DIR=/models auditor-stt:local

# GPU variant:
docker run --rm --gpus all -p 8090:8090 -v auditor-stt-models:/models \
  -e AUDITOR_STT_MODEL_DIR=/models auditor-stt:cuda
```

## Verify

```bash
# Wait for the model to finish downloading/loading:
curl http://localhost:8090/health
# {"status":"ok","model_id":"large-v3-turbo","device":"cpu","compute_type":"int8","loaded":true}

# Transcribe a fixture clip:
curl -F file=@fixture.wav -F language=fi http://localhost:8090/inference
```

## Compose

The root `docker-compose.yml` starts `auditor-stt` with a persistent model volume:

```bash
docker compose up -d
```

Then wait until:

```bash
curl http://localhost:8090/health
```

reports `"loaded":true`.
