# auditor-stt — CPU inference image
#
# faster-whisper (CTranslate2) inference server for Liturgos Auditor's self-hosted STT
# (docs/plans/plan_local_stt.md in live-captions-yt). No CUDA runtime — AUDITOR_STT_DEVICE=cpu is
# set explicitly so ModelHost doesn't waste startup time probing for a GPU
# that this image can never have; use Dockerfile.cuda for GPU inference.
#
# Model weights are NOT baked into the image — they download into
# AUDITOR_STT_MODEL_DIR (mount a volume there) on first boot.
#
# The service runs as the unprivileged user auditor (uid/gid 10001), never as root: it parses
# uploaded, untrusted audio (ffmpeg), so a bug there should not hand out root in the container.

FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN set -eux \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
       ffmpeg \
       ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# The code stays root-owned and read-only for the service user. The chmod makes it readable
# even if the build host checked the files out with a restrictive umask.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY auditor_stt ./auditor_stt
RUN pip install --no-cache-dir . \
  && chmod -R a+rX,go-w /app

# Everything below needs root, so it comes after apt/pip and before USER. The uid/gid are fixed
# numbers so a volume or bind mount can be chowned to a known id (see the upgrade notes in
# docker/liturgos-auditor-stt/README.md), and so `USER 10001:10001` below can be verified as
# non-root by orchestrators (Kubernetes runAsNonRoot cannot resolve a user name).
# The service writes only to the places created here: job state in /data, model downloads in
# /models, the fine-tuned model registry in /models/registry, temp files in /tmp (already
# world-writable) and, via HOME, huggingface_hub's own small state under ~/.cache.
# A named volume mounted on /data or /models copies this ownership when it is first created;
# volumes that already exist keep their old (root) owner and need a one-time chown.
# The home directory is mode 1777 (world-writable with the sticky bit, like /tmp) so that the image
# also works when the operator overrides the user, e.g. `--user "$(id -u):$(id -g)"` to write into a
# bind-mounted host directory: that uid has no passwd entry, and huggingface_hub would fail on a
# read-only ~/.cache. It holds only library caches, never data.
RUN set -eux \
  && groupadd --system --gid 10001 auditor \
  && useradd --system --uid 10001 --gid auditor --create-home --home-dir /home/auditor \
       --shell /usr/sbin/nologin auditor \
  && mkdir -p /data /models/registry \
  && chown -R 10001:10001 /data /models /home/auditor \
  && chmod 1777 /home/auditor

ENV HOME=/home/auditor \
    AUDITOR_STT_PORT=8090 \
    AUDITOR_STT_DEVICE=cpu \
    AUDITOR_STT_DEFAULT_LANGUAGE=fi \
    AUDITOR_STT_DATA_DIR=/data \
    AUDITOR_STT_REGISTRY_DIR=/models/registry

# Numeric form on purpose. The process starts as 10001; there is no entrypoint that starts
# as root and drops privileges afterwards.
USER 10001:10001

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
