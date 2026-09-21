"""Score Whisper models on a held-out split and on long-form audio.

A backend turns a model into `transcribe(samples) -> text` plus a `close()`
that frees it: models are loaded one at a time because the dev GPU has 8 GB.
`ct2_backend` is the SHIPPING path (the exported CTranslate2 model through
`serve.model.ModelHost`, exactly as the service runs it); `hf_backend` scores
a Hugging Face checkpoint and needs torch.

`longform_check` guards against the domain gap: the training data is short read
sentences without timestamp tokens, while the target is long sermon audio, so a
fine-tuned model can lose segment timing or long-form decoding even when it
wins on the test split. It runs candidate and baseline through the same batch
pipeline the service uses and compares WER and timestamp sanity.

Privacy: reference and hypothesis text never reach logs or return values.
Callers that want the predictions (`--dump-predictions`) pass a list in.
"""

import argparse
import gc
import io
import json
import logging
import os
from pathlib import Path

from ..serve.jobs.pipeline import transcribe_file
from ..serve.model import ModelHost
from .lineage import dataset_lineage
from .metrics import metrics_for, normalize_for_wer, per_speaker, wer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = "fi"
DEFAULT_BASELINE = "large-v3-turbo"
BACKENDS = ("ct2", "hf")

# faster-whisper decodes 30 s windows, so no segment is longer than that; a model
# that stopped emitting timestamp tokens produces window-long segments instead.
LONG_SEGMENT_SECONDS = 25.0
LONG_FRACTION_MIN_INCREASE = 0.05  # on top of "more than doubles", so 0 -> 1 long segment is not a regression
COVERAGE_MAX_DROP = 0.10
OVERLAP_TOLERANCE_SECONDS = 0.05  # segment times are rounded, so touching segments may overlap by a hair


def audio_array(audio, sampling_rate=SAMPLE_RATE):
    """Float samples at `sampling_rate` from an Audio cell read with decode=False.

    Same decoding as train.audio_array (datasets 4+ decodes through torchcodec,
    so the PyAV helper faster-whisper ships is used instead), kept here because
    importing train.py imports torch and CT2-only evaluation must not need it.
    """
    from faster_whisper.audio import decode_audio

    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    return decode_audio(source, sampling_rate=sampling_rate)


def _load_split(dataset_dir, split):
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(str(dataset_dir))
    if not isinstance(dataset, DatasetDict) or split not in dataset:
        raise ValueError(f"{dataset_dir} is not a dataset with a {split!r} split")
    return dataset[split]


def evaluate_dataset(transcribe, dataset_dir, split="test", *, speaker_ids=True, predictions=None):
    """Transcribe every utterance of `split` and score it. `transcribe(samples_float32_16k) -> str`.

    The result carries metrics only (see `metrics_for`) plus `split`,
    `dataset_version` (from build_metadata.json, None for older datasets),
    `audio_seconds` and, when `speaker_ids` and the dataset has a speaker_id
    column, the `per_speaker` spread. If `predictions` is a list, it receives
    one dict per utterance with recording_id, speaker_id, reference and
    hypothesis: the only place transcript text leaves this function.
    """
    from datasets import Audio

    rows = _load_split(dataset_dir, split).cast_column("audio", Audio(decode=False))
    with_speakers = speaker_ids and "speaker_id" in rows.column_names
    with_ids = "recording_id" in rows.column_names

    records = []
    for index, row in enumerate(rows, 1):
        samples = audio_array(row["audio"], SAMPLE_RATE)
        hypothesis = transcribe(samples) or ""
        speaker = row["speaker_id"] if with_speakers else None
        records.append({
            "speaker_id": speaker,
            "reference": row["text"],
            "hypothesis": hypothesis,
            "audio_seconds": len(samples) / SAMPLE_RATE,
        })
        if predictions is not None:
            predictions.append({
                "recording_id": row["recording_id"] if with_ids else None,
                "speaker_id": row["speaker_id"] if "speaker_id" in rows.column_names else None,
                "reference": row["text"],
                "hypothesis": hypothesis,
            })
        if index % 50 == 0:
            logger.info("Evaluated %d/%d utterances", index, len(rows))

    result = {
        "split": split,
        "dataset_version": dataset_lineage(dataset_dir)["dataset_version"],
        "audio_seconds": round(sum(r["audio_seconds"] for r in records), 3),
        **metrics_for((r["reference"], r["hypothesis"]) for r in records),
        "per_speaker": None,
    }
    if with_speakers:
        stats = per_speaker(records)
        del stats["speakers"]  # pseudonymous ids stay out of stored results
        result["per_speaker"] = stats
    return result


# --- backends -------------------------------------------------------------------------


def free_host(host):
    """Release a ModelHost's model (and its GPU memory) so the next one can load."""
    host.model = None
    host.loaded = False
    gc.collect()


# faster-whisper re-decodes a window that looks bad at a temperature above 0, which is random
# sampling from CTranslate2's own generator. Unseeded, the same model scored differently on the
# same split from one process to the next (a gate that can flip on a rerun), so the generator is
# seeded before each model loads. The decoding options themselves stay the service's.
EVAL_RANDOM_SEED = 1234


def load_ct2_host(model, device="auto", compute_type=None):
    """A loaded ModelHost for a CTranslate2 model directory or a faster-whisper alias.

    The download cache is the service's (AUDITOR_STT_MODEL_DIR), so a baseline
    alias already fetched for serving is not fetched again. CTranslate2's random
    generator is seeded first (EVAL_RANDOM_SEED), so scores are repeatable.
    """
    path = Path(model)
    if path.is_dir():
        if not (path / "model.bin").is_file():
            raise ValueError(
                f"{path} is not a CTranslate2 model directory (no model.bin). A Hugging Face checkpoint "
                "needs --backend hf, or `auditor-stt export` first"
            )
        model = str(path)
    host = ModelHost(
        model_id=str(model),
        model_dir=os.environ.get("AUDITOR_STT_MODEL_DIR"),
        device=device,
        compute_type=compute_type,
    )
    import ctranslate2

    ctranslate2.set_random_seed(EVAL_RANDOM_SEED)
    host.load()
    return host


def ct2_backend(model, device="auto", compute_type=None, language=DEFAULT_LANGUAGE):
    """(transcribe, close) for the shipping path: ModelHost.transcribe_array on a CT2 model."""
    host = load_ct2_host(model, device, compute_type)

    def transcribe(samples):
        return host.transcribe_array(samples, language, word_timestamps=False)["text"]

    transcribe.device, transcribe.compute_type = host.device, host.compute_type
    return transcribe, lambda: free_host(host)


def hf_backend(model_dir, language=DEFAULT_LANGUAGE, device="auto", max_new_tokens=225):
    """(transcribe, close) for a full Hugging Face Whisper checkpoint. Needs torch."""
    model_dir = Path(model_dir)
    if (model_dir / "adapter_config.json").is_file():
        raise ValueError(
            f"{model_dir} is a LoRA adapter, not a model. Evaluate the merged checkpoint instead: "
            "`auditor-stt export --merge-lora` leaves it in <export>/_merged_hf next to the CT2 model "
            "(which is what --backend ct2 scores)"
        )
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from .compat import fp32_load_kwargs

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(str(model_dir), language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), **fp32_load_kwargs()).to(device).eval()
    # A language can only be forced on a checkpoint that has the language table.
    options = {"language": language, "task": "transcribe"} if getattr(model.generation_config, "lang_to_id", None) else {}

    def transcribe(samples):
        features = processor.feature_extractor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(
                input_features=features.input_features.to(device), max_new_tokens=max_new_tokens, **options
            )
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def close():
        nonlocal model
        model = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    transcribe.device = device
    return transcribe, close


def open_backend(backend, model, device="auto", compute_type=None, language=DEFAULT_LANGUAGE):
    if backend == "ct2":
        return ct2_backend(model, device, compute_type, language)
    if backend == "hf":
        return hf_backend(model, language, device)
    raise ValueError(f"backend must be one of {list(BACKENDS)}, got {backend!r}")


def evaluate_model(model, dataset_dir, split="test", *, backend="ct2", device="auto", compute_type=None,
                   language=DEFAULT_LANGUAGE, predictions=None):
    """Load `model`, score it on the split, free it. See evaluate_dataset for the result."""
    _load_split(dataset_dir, split)  # a bad dataset should fail before a large model is loaded
    transcribe, close = open_backend(backend, model, device, compute_type, language)
    try:
        return evaluate_dataset(transcribe, dataset_dir, split, predictions=predictions)
    finally:
        close()


# --- long-form regression check ----------------------------------------------------------


def read_reference(path):
    """The normalised reference words of a UTF-8 text file."""
    words = normalize_for_wer(Path(path).read_text(encoding="utf-8-sig"))
    if not words:
        raise ValueError(f"The long-form reference {path} contains no words")
    return words


def timestamp_sanity(segments, duration_seconds, long_seconds=LONG_SEGMENT_SECONDS):
    """Whether segment timing looks like a transcript's: ordered, short, and reaching the end.

    `order_violations` counts segments that end before they start or start before
    the previous one ended; `long_segment_fraction` is the share lasting at least
    `long_seconds`; `coverage` is where the last segment ends relative to the audio.
    """
    violations, previous_end = 0, 0.0
    for segment in segments:
        if segment["end"] < segment["start"] or segment["start"] < previous_end - OVERLAP_TOLERANCE_SECONDS:
            violations += 1
        previous_end = segment["end"]
    n = len(segments)
    long_segments = sum(1 for s in segments if s["end"] - s["start"] >= long_seconds)
    has_audio = duration_seconds > 0
    return {
        "n_segments": n,
        "order_violations": violations,
        "long_segment_fraction": round(long_segments / n, 6) if n else 0.0,
        "long_segment_seconds": long_seconds,
        "coverage": round(segments[-1]["end"] / duration_seconds, 4) if n and has_audio else 0.0,
        "segments_per_minute": round(n / (duration_seconds / 60), 3) if has_audio else 0.0,
    }


def compare_longform(candidate, baseline, tolerance):
    """(regression, reasons): the candidate is worse than the baseline beyond `tolerance` or on timing sanity.

    Timing is a regression when there are more order violations than the
    baseline's, when the share of window-long segments more than doubles (and
    grows by at least LONG_FRACTION_MIN_INCREASE), or when the last segment ends
    COVERAGE_MAX_DROP earlier relative to the audio.
    """
    reasons = []
    delta = round(candidate["wer"] - baseline["wer"], 6)
    if delta > tolerance:
        reasons.append(
            f"WER {candidate['wer']:.4f} is {delta:.4f} above the baseline's {baseline['wer']:.4f} "
            f"(tolerance {tolerance:.4f})"
        )
    cand, base = candidate["sanity"], baseline["sanity"]
    if cand["order_violations"] > base["order_violations"]:
        reasons.append(
            f"{cand['order_violations']} out-of-order or overlapping segments, baseline {base['order_violations']}"
        )
    grown = cand["long_segment_fraction"] - base["long_segment_fraction"]
    if cand["long_segment_fraction"] > 2 * base["long_segment_fraction"] and grown >= LONG_FRACTION_MIN_INCREASE:
        reasons.append(
            f"window-long segments went from {base['long_segment_fraction']:.1%} to {cand['long_segment_fraction']:.1%}"
        )
    if base["coverage"] - cand["coverage"] > COVERAGE_MAX_DROP:
        reasons.append(f"coverage fell from {base['coverage']:.1%} to {cand['coverage']:.1%}")
    return bool(reasons), reasons


def _longform_run(model, audio_path, reference, *, workdir, chunk_seconds, language, device, compute_type):
    """Transcribe the audio with one model, freed afterwards; keeps the metrics, never the text."""
    host = load_ct2_host(model, device, compute_type)
    try:
        result = transcribe_file(host, audio_path, workdir=workdir, language=language, chunk_seconds=chunk_seconds)
    finally:
        free_host(host)
    hypothesis = normalize_for_wer(result["text"])
    return {
        "wer": wer([reference], [hypothesis]),
        "words": len(hypothesis.split()),
        "sanity": timestamp_sanity(result["segments"], result["duration_seconds"]),
    }, result["duration_seconds"]


def longform_check(candidate, baseline, audio_path, reference_path, *, tolerance=0.02, chunk_seconds=60.0,
                   workdir, language=DEFAULT_LANGUAGE, device="auto", compute_type=None):
    """Run both CT2 models over the same long audio and compare them to a hand-corrected reference.

    Uses the service's batch pipeline (`transcribe_file`: 60 s chunks snapped to
    pauses, prompt carried between chunks). The models run one after the other,
    the first freed before the second loads. `workdir` holds the normalised WAV
    while a model works. Nothing textual is returned or written.

    Returns the per-model `wer` (normalised), `words` and `sanity`, `wer_delta`,
    and `regression` with the `reasons` for it.
    """
    reference = read_reference(reference_path)
    options = dict(workdir=workdir, chunk_seconds=chunk_seconds, language=language, device=device,
                   compute_type=compute_type)
    candidate_run, duration = _longform_run(candidate, audio_path, reference, **options)
    baseline_run, _ = _longform_run(baseline, audio_path, reference, **options)
    if baseline_run["sanity"]["n_segments"] == 0:
        # Both models saying nothing would look like "no regression" while comparing nothing.
        raise ValueError("The baseline model found no speech in the long-form audio, so there is nothing to compare against")
    regression, reasons = compare_longform(candidate_run, baseline_run, tolerance)
    return {
        "tolerance": tolerance,
        "chunk_seconds": chunk_seconds,
        "audio_seconds": round(duration, 3),
        "reference_words": len(reference.split()),
        "candidate": candidate_run,
        "baseline": baseline_run,
        "wer_delta": round(candidate_run["wer"] - baseline_run["wer"], 6),
        "regression": regression,
        "reasons": reasons,
    }


# --- legacy entry point ------------------------------------------------------------------


def evaluate(model_dir, dataset_dir, batch_size=4, language=DEFAULT_LANGUAGE):
    """What scripts/evaluate-whisper.py always did: a Hugging Face checkpoint on the test split.

    Writes test_metrics.json into `model_dir`. `wer` keeps its old meaning (raw,
    case and punctuation sensitive); the normalised numbers are new. `batch_size`
    is accepted for compatibility: utterances are decoded one at a time.
    """
    transcribe, close = hf_backend(model_dir, language)
    try:
        scored = evaluate_dataset(transcribe, dataset_dir, "test")
        device = transcribe.device
    finally:
        close()
    result = {
        "model": str(Path(model_dir).resolve()),
        "dataset": str(Path(dataset_dir).resolve()),
        "test_examples": scored["n"],
        "wer": scored["wer_raw"],
        "device": device,
        "wer_raw": scored["wer_raw"],
        "cer_raw": scored["cer_raw"],
        "wer_normalised": scored["wer_normalised"],
        "cer_normalised": scored["cer_normalised"],
        "dataset_version": scored["dataset_version"],
        "per_speaker": scored["per_speaker"],
    }
    (Path(model_dir) / "test_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a Whisper checkpoint on Auditor's test split (see `auditor-stt eval` for the full gate)"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--batch-size", type=int, default=4, help="Ignored; kept so existing invocations keep working")
    p.add_argument("--language", default=DEFAULT_LANGUAGE)
    args = p.parse_args(argv)
    result = evaluate(args.model, args.dataset, batch_size=args.batch_size, language=args.language)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
