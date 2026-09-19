# liturgos-auditor

Self-hosted, trainable Finnish speech-to-text.

This repository contains the standalone speech-to-text service extracted from
the original `lcyt-stt` implementation in `live-captions-yt`.

## Components

- FastAPI inference service using faster-whisper / CTranslate2
- Whisper-compatible `POST /inference` endpoint
- `GET /health` model/device status endpoint
- bounded single-worker inference queue
- CPU and CUDA Docker images
- crowd-source-voice dataset snapshot and build pipeline
- speaker-disjoint train/dev/test splitting when speaker IDs are available
- Finnish text normalization

The service is deliberately usable independently of any one consumer. Live
Captions YT and other applications can consume the HTTP inference API.

## Development

```bash
cd python-packages/liturgos-auditor-stt
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run the service:

```bash
auditor-stt serve --port 8090
```

See `python-packages/liturgos-auditor-stt/README.md` for configuration and
dataset pipeline documentation.
