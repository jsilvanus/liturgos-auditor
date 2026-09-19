# faster-whisper model download

The STT service uses faster-whisper and defaults to the built-in `large-v3-turbo` model alias. Current faster-whisper resolves that alias to its CTranslate2 large-v3-turbo model on Hugging Face.

The model is downloaded automatically the first time `WhisperModel` loads it. You can also download/warm it explicitly before starting the service.

## Download with the repository helper

From the repository root:

```bash
python3 scripts/download-whisper-model.py
```

Use another model:

```bash
python3 scripts/download-whisper-model.py large-v3
```

The helper also respects the same environment variables as the service:

```bash
export AUDITOR_STT_MODEL=large-v3-turbo
export AUDITOR_STT_MODEL_DIR=/srv/models
python3 scripts/download-whisper-model.py
```

The model is constructed with CPU/int8 for the download command, so CUDA is not required.

## Docker

For a container deployment, set `AUDITOR_STT_MODEL_DIR` to a persistent volume if you do not want the model downloaded again when the container is recreated.

For example, conceptually:

```yaml
environment:
  AUDITOR_STT_MODEL_DIR: /models
volumes:
  - auditor-stt-models:/models
```

The service itself uses `AUDITOR_STT_MODEL` and `AUDITOR_STT_MODEL_DIR` when constructing `WhisperModel`.

## What is actually downloaded?

This is the CTranslate2 model used by faster-whisper, not an original OpenAI Whisper checkpoint. The `large-v3-turbo` alias resolves to the CTranslate2 conversion used by faster-whisper.

The service's `/health` endpoint reports the configured model ID and whether it has finished loading.
