# auditor-stt — CPU training image
#
# Whisper fine-tuning pipeline (dataset build, training, LoRA export) with PyTorch CPU.
# PyTorch is excluded from the serving image (docker/liturgos-auditor-stt/Dockerfile)
# because native torch is blocked by Windows Smart App Control on the maintainer's machine;
# training runs in this Linux container instead.
#
# Base model weights and training data are NOT baked into the image — they download on demand
# into HF_HOME (/hf, mount a volume) and AUDITOR_STT_TRAIN_DATA_DIR (/data, mount a volume).
#
# The commands run as the unprivileged user auditor (uid/gid 10001), never as root: they handle
# voice recordings, and nothing here needs privileges. See the user block below for what that
# means for mounted volumes.

FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_COMMIT=

RUN set -eux \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
       ffmpeg \
       ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first from the PyTorch index so pip doesn't pull the huge default (CUDA) build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY auditor_stt ./auditor_stt

# Install the package with the [training] extra.
# torch is already present, so pip will reuse it.
# The code stays root-owned and read-only for the service user. The chmod makes it readable
# even if the build host checked the files out with a restrictive umask.
RUN pip install --no-cache-dir ".[training]" \
  && chmod -R a+rX,go-w /app

# Everything below needs root, so it comes after apt/pip and before USER. The uid/gid are fixed
# numbers so a volume or bind mount can be chowned to a known id, and so `USER 10001:10001` below
# can be verified as non-root by orchestrators (Kubernetes runAsNonRoot cannot resolve a user name).
# The commands write only to the places created here: /data (ledger, audio, datasets), /models
# (runs, exports, registry), /hf (HuggingFace cache; HF_HOME) and the home directory (HOME, for
# libraries that keep state under ~/.cache). A named volume mounted on one of them copies this
# ownership when it is first created; volumes that already exist keep their old (root) owner and
# need a one-time chown (see "Existing volumes" in docker/liturgos-auditor-train/README.md).
# /hf and the home directory are mode 1777 (world-writable with the sticky bit, like /tmp) so that
# the image also works when the operator overrides the user, e.g. `--user "$(id -u):$(id -g)"` to
# write into a bind-mounted host directory: that uid has no passwd entry and is not 10001, and
# HuggingFace downloads (into /hf) and library caches (under ~/.cache) would fail on directories
# that only 10001 can write. They hold public model weights and caches, never recordings.
# /data and /models stay 755: they hold voice recordings and models trained on them, so only their
# owner should write there.
RUN set -eux \
  && groupadd --system --gid 10001 auditor \
  && useradd --system --uid 10001 --gid auditor --create-home --home-dir /home/auditor \
       --shell /usr/sbin/nologin auditor \
  && mkdir -p /data /models /hf \
  && chown -R 10001:10001 /data /models /home/auditor \
  && chown 10001:10001 /hf \
  && chmod 1777 /hf /home/auditor

# HOME is set explicitly because an arbitrary --user has no passwd entry and Docker would then
# give it HOME=/.
ENV HOME=/home/auditor \
    HF_HOME=/hf \
    AUDITOR_STT_TRAIN_DATA_DIR=/data \
    AUDITOR_STT_GIT_COMMIT=$GIT_COMMIT \
    AUDITOR_STT_REGISTRY_DIR=/models/registry

# Numeric form on purpose. The process starts as 10001; there is no entrypoint that starts
# as root and drops privileges afterwards.
USER 10001:10001

ENTRYPOINT ["auditor-stt"]

# Build (from repo root — build context is python-packages/liturgos-auditor-stt):
#   docker build -t auditor-train:cpu -f docker/liturgos-auditor-train/Dockerfile \
#     --build-arg GIT_COMMIT=$(git rev-parse HEAD) python-packages/liturgos-auditor-stt
#
# Run examples (mount data dir, models dir, and HuggingFace cache volume):
#   docker run --rm -v <host data dir>:/data -v <host models dir>:/models -v auditor-hf:/hf \
#     -e CSV_ADMIN_TOKEN auditor-train:cpu dataset sync --base-url <url> --corpus-id <id>
#   docker run --rm -v <host data dir>:/data -v <host models dir>:/models -v auditor-hf:/hf \
#     auditor-train:cpu dataset build --corpus-id <id> --out /models
#   docker run --rm -v <host data dir>:/data -v <host models dir>:/models -v auditor-hf:/hf \
#     auditor-train:cpu train --dataset /models/<version> --out /models/run --peft lora --gradient-checkpointing
#   docker run --rm -v <host models dir>:/models -v auditor-hf:/hf \
#     auditor-train:cpu export --model /models/run/checkpoint-X --out /models/exported --merge-lora
#
# On a Linux host a bind-mounted <host dir> must be writable by uid 10001, or add
# --user "$(id -u):$(id -g)" to the docker run to write as yourself.
