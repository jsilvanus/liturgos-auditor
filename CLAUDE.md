# CLAUDE.md

## Never store model weights in this repository

This repository is public. Do not commit model weights or anything derived from them: no Whisper or faster-whisper/CTranslate2 files, no fine-tuned or LoRA checkpoints, no exported models, no registry contents (`models/`, `models/registry`), and no training data or recordings. Weights belong somewhere else (a separate repository for privately trained models, or external storage); this repository holds code only.

- Models are downloaded at run time into `AUDITOR_STT_MODEL_DIR` or the registry directory, outside the source tree.
- If a task seems to need a weight file in the repo, stop and ask instead.
- Model licenses differ from the MIT code license; see the License section of `README.md` and `NOTICE`.

## Live sessions (pull type)

`/v1/live` (code in `python-packages/liturgos-auditor-stt/auditor_stt/serve/live/`, docs in `docs/live-sessions.md`) pulls an rtsp/srt stream through an fffleet stream job (or a local ffmpeg), segments it and publishes final transcripts as Server-Sent Events. Per-session options: `language`, `prompt`, `client_ref`, `delay_seconds` (holds each transcript until `wall_end + delay`; times in the event stay those of the source). Settings are `AUDITOR_STT_LIVE_*` and `AUDITOR_STT_MAX_LIVE_SESSIONS` (table in `README.md`). When you change the routes, events or settings, update `docs/live-sessions.md` and the `README.md` table together. Tests: `tests/test_live.py`, `tests/test_live_upload_cap.py`; never run against a real MediaMTX, fleet or GPU. Never commit real service recordings or transcripts with personal names.
