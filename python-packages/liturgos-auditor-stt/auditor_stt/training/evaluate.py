"""Evaluate a fine-tuned Whisper checkpoint on the held-out test split."""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from jiwer import wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def evaluate(model_dir, dataset_dir, batch_size=4, language="fi"):
    dataset = load_from_disk(dataset_dir)
    if "test" not in dataset:
        raise ValueError("Dataset must contain a test split")

    processor = WhisperProcessor.from_pretrained(model_dir, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    references = []
    hypotheses = []
    for start in range(0, len(dataset["test"]), batch_size):
        batch = dataset["test"][start:start + batch_size]
        features = []
        for audio in batch["audio"]:
            encoded = processor.feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            )
            features.append(encoded.input_features[0])
        input_features = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.inference_mode():
            predicted = model.generate(input_features=input_features, max_new_tokens=225)
        decoded = processor.batch_decode(predicted, skip_special_tokens=True)
        references.extend(batch["text"])
        hypotheses.extend(decoded)

    score = wer(references, hypotheses)
    result = {
        "model": str(Path(model_dir).resolve()),
        "dataset": str(Path(dataset_dir).resolve()),
        "test_examples": len(references),
        "wer": score,
        "device": device,
    }
    output = Path(model_dir) / "test_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a Whisper checkpoint on Auditor's test split")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--language", default="fi")
    args = p.parse_args(argv)
    result = evaluate(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
