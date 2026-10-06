# CLAUDE.md

## Never store model weights in this repository

This repository is public. Do not commit model weights or anything derived from them: no Whisper or faster-whisper/CTranslate2 files, no fine-tuned or LoRA checkpoints, no exported models, no registry contents (`models/`, `models/registry`), and no training data or recordings. Weights belong somewhere else (a separate repository for privately trained models, or external storage); this repository holds code only.

- Models are downloaded at run time into `AUDITOR_STT_MODEL_DIR` or the registry directory, outside the source tree.
- If a task seems to need a weight file in the repo, stop and ask instead.
- Model licenses differ from the MIT code license; see the License section of `README.md` and `NOTICE`.
