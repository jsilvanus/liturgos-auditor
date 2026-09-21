# auditor-stt — CPU training image
#
# Whisper fine-tuning pipeline (dataset build, training, LoRA export) with PyTorch CPU.
# PyTorch is excluded from the serving image (docker/liturgos-auditor-stt/Dockerfile)
# because native torch is blocked by Windows Smart App Control on the maintainer's machine;
# training runs in this Linux container instead.
#
# Base model weights and training data are NOT baked into the image — they download on demand
# into HF_HOME (/hf, mount a volume) and AUDITOR_STT_TRAIN_DATA_DIR (/data, mount a volume).

FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

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
RUN pip install --no-cache-dir ".[training]"

RUN mkdir -p /data /models /hf

ENV HF_HOME=/hf \
    AUDITOR_STT_TRAIN_DATA_DIR=/data

ENTRYPOINT ["auditor-stt"]

# Build (from repo root — build context is python-packages/liturgos-auditor-stt):
#   docker build -t auditor-train:cpu -f docker/liturgos-auditor-train/Dockerfile python-packages/liturgos-auditor-stt
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
