# auditor-stt — CPU inference image
#
# faster-whisper (CTranslate2) inference server for Liturgos Auditor's self-hosted STT
# (docs/plans/plan_local_stt.md in live-captions-yt). No CUDA runtime — AUDITOR_STT_DEVICE=cpu is
# set explicitly so ModelHost doesn't waste startup time probing for a GPU
# that this image can never have; use Dockerfile.cuda for GPU inference.
#
# Model weights are NOT baked into the image — they download into
# AUDITOR_STT_MODEL_DIR (mount a volume there) on first boot.

FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN set -eux \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
       ffmpeg \
       ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY auditor_stt ./auditor_stt
RUN pip install --no-cache-dir .

RUN mkdir -p /data /models

ENV AUDITOR_STT_PORT=8090 \
    AUDITOR_STT_DEVICE=cpu \
    AUDITOR_STT_DEFAULT_LANGUAGE=fi \
    AUDITOR_STT_DATA_DIR=/data \
    AUDITOR_STT_REGISTRY_DIR=/models/registry

EXPOSE 8090

CMD ["auditor-stt", "serve"]

# Build (local dev, from repo root — build context is the package source,
# not this docker/ directory):
#   docker build -t auditor-stt:local -f docker/liturgos-auditor-stt/Dockerfile python-packages/liturgos-auditor-stt
# Run:
#   docker run --rm -p 8090:8090 -v auditor-stt-models:/models \
#     -e AUDITOR_STT_MODEL_DIR=/models -v auditor-stt-data:/data \
#     -e AUDITOR_STT_DATA_DIR=/data auditor-stt:local
# Verify:
#   curl http://localhost:8090/health
