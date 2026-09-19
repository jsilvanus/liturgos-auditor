# Auditor STT training pipeline

The repository now has an end-to-end training path:

1. Pull a validated text corpus from crowd-source-voice.
2. Build a Hugging Face DatasetDict with train/dev/test splits.
3. Fine-tune an OpenAI Whisper checkpoint with the local training command.
4. Evaluate the trained checkpoint on the held-out test split with WER.
5. Convert the trained Transformers checkpoint to CTranslate2.
6. Serve the converted model with the existing auditor-stt faster-whisper service.

## Dataset

Pull a snapshot:

    auditor-stt dataset pull --base-url https://crowd-source-voice.example --corpus-id 123 --out data/snapshots/corpus-123

Build the dataset:

    auditor-stt dataset build --snapshot data/snapshots/corpus-123 --out data/datasets/corpus-123

The builder uses speaker-disjoint splits whenever every recording has a speaker_id. Otherwise it explicitly marks the dataset as non-speaker-disjoint and uses a deterministic utterance-level split.

## Fine-tuning

Install the training dependencies:

    pip install -e '.[training]'

Then:

    python scripts/train-whisper.py --dataset data/datasets/corpus-123 --output models/whisper-fi-v1 --model openai/whisper-large-v3-turbo --language fi --batch-size 4 --gradient-accumulation-steps 4 --fp16

The defaults are intentionally conservative. Adjust batch size and accumulation to the available GPU memory.

The resulting directory is a normal Transformers checkpoint and includes training_metadata.json.

## Held-out evaluation

Run this only after training:

    python scripts/evaluate-whisper.py --model models/whisper-fi-v1 --dataset data/datasets/corpus-123

The command writes test_metrics.json with the test-set word error rate (WER). The test split is never passed to the training loop.

## Export for inference

The serving image uses faster-whisper/CTranslate2, not the Transformers checkpoint directly:

    python scripts/export-whisper-ct2.py --model models/whisper-fi-v1 --output models/whisper-fi-v1-ct2 --quantization float16

Point the service at the exported directory:

    AUDITOR_STT_MODEL=/models/whisper-fi-v1-ct2 docker compose up -d

## Dependency boundary

Training and inference have deliberately separate dependencies. The inference container does not install PyTorch/Transformers, so production STT remains a small serving image.

The test split is reserved for final held-out evaluation and is not used by the fine-tuning trainer.
