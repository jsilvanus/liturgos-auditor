"""Fine-tune a Whisper checkpoint on an Auditor Hugging Face dataset.

The serving stack remains faster-whisper/CTranslate2; this training command uses
the Transformers Whisper implementation and produces a normal Hugging Face
checkpoint, or with --peft lora a small adapter that export_ct2.py merges into
its base model. Run export_ct2.py afterwards to create the directory consumed by
AUDITOR_STT_MODEL.
"""

import argparse
import io
import json
import logging
from pathlib import Path

import torch
from datasets import Audio, load_from_disk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from ..dataset.normalize import normalize_text
from .compat import fp32_load_kwargs, save_processor, warmup_kwargs
from .lineage import build_metadata
from .preflight import PRESETS, check_gpu_memory, resolve_model_id

logger = logging.getLogger(__name__)


def _load_dataset(path):
    dataset = load_from_disk(path)
    if not {"train", "dev", "test"} <= set(dataset.keys()):
        raise ValueError("Dataset must contain train, dev and test splits")
    return dataset


def audio_array(audio, sampling_rate=16000):
    """Float samples at `sampling_rate` from an Audio cell read with decode=False.

    datasets 4+ decodes through torchcodec and returns decoder objects instead of
    the {"array": ...} dicts older versions gave, so decode with the PyAV helper
    faster-whisper already ships and stay independent of the datasets version.
    """
    from faster_whisper.audio import decode_audio

    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    return decode_audio(source, sampling_rate=sampling_rate)


def _prepare_dataset(dataset, processor):
    sampling_rate = processor.feature_extractor.sampling_rate

    def prepare(batch):
        array = audio_array(batch["audio"], sampling_rate)
        inputs = processor.feature_extractor(array, sampling_rate=sampling_rate)
        labels = processor.tokenizer(
            normalize_text(batch["text"]),
            max_length=processor.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        return {"input_features": inputs.input_features[0], "labels": labels}

    return dataset.cast_column("audio", Audio(decode=False)).map(
        prepare,
        remove_columns=dataset["train"].column_names,
        desc="Preparing Whisper features",
    )


def _apply_lora(model, r, alpha, dropout, target_modules, gradient_checkpointing):
    """Wrap `model` with LoRA adapters; only the adapters stay trainable."""
    from peft import LoraConfig, get_peft_model

    if gradient_checkpointing:
        # The frozen inputs do not require grad, and reentrant checkpointing
        # would then give the adapters inside a checkpointed block no gradient.
        model.enable_input_require_grads()
    peft_model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(target_modules), bias="none",
    ))
    # fp16 mixed precision refuses to unscale fp16 gradients, so adapters must be fp32.
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return peft_model


def train(dataset_dir, output_dir, model_id=None, language="fi", task="transcribe",
          learning_rate=1e-5, epochs=3.0, batch_size=4, gradient_accumulation_steps=1,
          eval_batch_size=None, warmup_ratio=0.05, seed=42, fp16=False, bf16=False,
          preset=None, peft="none", lora_r=32, lora_alpha=64, lora_dropout=0.05,
          lora_target_modules=("q_proj", "v_proj"), gradient_checkpointing=False,
          skip_preflight=False):
    if peft not in ("none", "lora"):
        raise ValueError(f"peft must be 'none' or 'lora', got {peft!r}")
    model_id = resolve_model_id(model_id, preset)
    eval_batch_size = eval_batch_size or batch_size
    dataset = _load_dataset(dataset_dir)

    if not skip_preflight:
        check_gpu_memory(
            model_id, "lora" if peft == "lora" else "full", fp16 or bf16,
            gradient_checkpointing, batch_size,
        )

    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, **fp32_load_kwargs())
    base_model_revision = getattr(model.config, "_commit_hash", None)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    peft_info = {"method": peft}
    base_model = model
    if peft == "lora":
        model = _apply_lora(base_model, lora_r, lora_alpha, lora_dropout, lora_target_modules, gradient_checkpointing)
        trainable, total = model.get_nb_trainable_parameters()
        peft_info.update({
            "r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
            "target_modules": list(lora_target_modules),
            "trainable_parameters": trainable, "total_parameters": total,
        })
        logger.info("LoRA: training %d of %d parameters (%.2f%%)", trainable, total, 100 * trainable / total)

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

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        # A PeftModel's forward takes **kwargs, so the Trainer cannot infer which
        # dataset columns and label names the model uses.
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        save_total_limit=2,
        **warmup_kwargs(warmup_ratio),
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
    trainer.save_model(str(output_dir))  # the adapter alone for a PeftModel
    save_processor(processor, output_dir)
    if peft == "lora":
        # An adapter dir carries no generation config; export merges it back so
        # the merged model keeps the forced language and task.
        base_model.generation_config.save_pretrained(str(output_dir))

    config = {
        "language": language, "task": task, "learning_rate": learning_rate, "epochs": epochs,
        "batch_size": batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_batch_size": eval_batch_size, "warmup_ratio": warmup_ratio, "seed": seed,
        "fp16": fp16, "bf16": bf16, "gradient_checkpointing": gradient_checkpointing,
    }
    metadata = build_metadata(
        model_id=model_id, preset=preset, base_model_revision=base_model_revision,
        dataset_dir=dataset_dir, config=config, peft=peft_info,
        sizes={name: len(prepared[name]) for name in ("train", "dev", "test")},
        train_metrics=result.metrics,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
    )
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune Whisper on an Auditor dataset")
    p.add_argument("--dataset", required=True, dest="dataset_dir")
    p.add_argument("--output", required=True, dest="output_dir")
    p.add_argument("--model", default=None, dest="model_id",
                   help="Base model id or path; overrides --preset (default: preset large-v3-turbo)")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
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
    p.add_argument("--peft", choices=["none", "lora"], default="none")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args(argv)
    return train(**vars(args))


if __name__ == "__main__":
    main()
