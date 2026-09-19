"""Fine-tune a Whisper checkpoint on an Auditor Hugging Face dataset.

The serving stack remains faster-whisper/CTranslate2; this training command uses
the Transformers Whisper implementation and produces a normal Hugging Face
checkpoint. Run export_ct2.py afterwards to create the directory consumed by
AUDITOR_STT_MODEL.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from ..dataset.normalize import normalize_text


def _load_dataset(path):
    dataset = load_from_disk(path)
    if not {"train", "dev", "test"} <= set(dataset.keys()):
        raise ValueError("Dataset must contain train, dev and test splits")
    return dataset


def _prepare_dataset(dataset, processor):
    def prepare(batch):
        audio = batch["audio"]
        inputs = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])
        labels = processor.tokenizer(
            normalize_text(batch["text"]),
            max_length=processor.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        return {"input_features": inputs.input_features[0], "labels": labels}

    return dataset.map(
        prepare,
        remove_columns=dataset["train"].column_names,
        desc="Preparing Whisper features",
    )


def train(dataset_dir, output_dir, model_id="openai/whisper-large-v3-turbo",
          language="fi", task="transcribe", learning_rate=1e-5, epochs=3.0,
          batch_size=4, gradient_accumulation_steps=1, eval_batch_size=None,
          warmup_ratio=0.05, seed=42, fp16=False, bf16=False):
    dataset = _load_dataset(dataset_dir)
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    prepared = _prepare_dataset(dataset, processor)

    class DataCollator:
        def __call__(self, features):
            input_features = torch.tensor(
                [f["input_features"] for f in features], dtype=torch.float32
            )
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels = processor.tokenizer.pad(label_features, return_tensors="pt").input_ids
            labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
            if labels.shape[1] > 0 and (labels[:, 0] == processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]
            return {"input_features": input_features, "labels": labels}

    eval_batch_size = eval_batch_size or batch_size
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        fp16=fp16,
        bf16=bf16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=prepared["train"],
        eval_dataset=prepared["dev"],
        data_collator=DataCollator(),
        processing_class=processor,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metadata = {
        "base_model": model_id,
        "language": language,
        "task": task,
        "dataset": str(Path(dataset_dir).resolve()),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "train_examples": len(prepared["train"]),
        "dev_examples": len(prepared["dev"]),
        "test_examples": len(prepared["test"]),
        "train_metrics": result.metrics,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
    )
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune Whisper on an Auditor dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--language", default="fi")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args(argv)
    return train(**vars(args))


if __name__ == "__main__":
    main()
