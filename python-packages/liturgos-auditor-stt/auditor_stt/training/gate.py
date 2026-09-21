"""The evaluation gate: may this fine-tuned model replace the one in service?

A candidate passes when it beats the zero-shot baseline on the held-out test
split (normalised WER, same split, same decoding path) and, when long-form
material is supplied, does not regress on it. The verdict is written to
`gate.json` next to the candidate, where `models promote` reads it (`passed`
and `dataset_version`). A failed gate is still written: it is the record of why.

The file holds metrics and verdicts only, never transcripts, and no speaker
ids: the per-speaker part is a spread (min, median, p90, max).
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import (
    DEFAULT_BASELINE,
    DEFAULT_LANGUAGE,
    evaluate_model,
    longform_check,
    read_reference,
)

logger = logging.getLogger(__name__)

GATE_FILENAME = "gate.json"
GATE_SCHEMA = 1
DEFAULT_LONGFORM_TOLERANCE = 0.02
DEFAULT_LONGFORM_CHUNK_SECONDS = 60.0

BEATS_BASELINE = "beats_baseline_on_test"
LONGFORM_NO_REGRESSION = "longform_no_regression"


def gate_path(candidate, out=None):
    """Where the gate is written: `out`, else gate.json inside the candidate directory."""
    if out is not None:
        return Path(out)
    if not Path(candidate).is_dir():
        raise ValueError(f"{candidate} is not a directory, so there is nowhere to put {GATE_FILENAME}: pass an explicit output path")
    return Path(candidate) / GATE_FILENAME


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check(name, passed, detail):
    return {"name": name, "passed": passed, "detail": detail}


def _beats_baseline_check(candidate, baseline, min_improvement, split):
    c, b = candidate["wer_normalised"], baseline["wer_normalised"]
    if c is None or b is None:
        return _check(BEATS_BASELINE, False, f"normalised WER is undefined: the {split} split has no reference words")
    limit = round(b * (1 - min_improvement), 6)  # the WERs are rounded too; float noise must not decide a tie
    return _check(
        BEATS_BASELINE,
        bool(c < limit),
        f"candidate normalised WER {c:.4f} vs baseline {b:.4f} on {split}; must be below {limit:.4f}",
    )


def _longform_check_entry(longform, required):
    if longform is None:
        if required:
            return _check(LONGFORM_NO_REGRESSION, False, "long-form audio and reference are required but were not given")
        return _check(LONGFORM_NO_REGRESSION, None, "skipped: no long-form audio and reference given")
    c, b = longform["candidate"]["wer"], longform["baseline"]["wer"]
    verdict = "; ".join(longform["reasons"]) if longform["regression"] else "timestamp sanity holds"
    return _check(
        LONGFORM_NO_REGRESSION,
        not longform["regression"],
        f"candidate WER {c:.4f} vs baseline {b:.4f} (tolerance {longform['tolerance']:.4f}); {verdict}",
    )


def _relative_improvement(candidate, baseline):
    c, b = candidate["wer_normalised"], baseline["wer_normalised"]
    if c is None or not b:
        return None
    return round((b - c) / b, 6)


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def evaluate_gate(
    candidate,
    dataset_dir,
    *,
    split="test",
    baseline=DEFAULT_BASELINE,
    backend="ct2",
    device="auto",
    compute_type=None,
    language=DEFAULT_LANGUAGE,
    longform_audio=None,
    longform_reference=None,
    longform_tolerance=DEFAULT_LONGFORM_TOLERANCE,
    longform_chunk_seconds=DEFAULT_LONGFORM_CHUNK_SECONDS,
    min_improvement=0.0,
    require_longform=False,
    out=None,
    predictions=None,
):
    """Score candidate and baseline on the split (and on long-form audio if given), write and return the gate.

    `candidate` is a CT2 model directory (or, with backend="hf", a Hugging Face
    checkpoint); the baseline is always scored through CT2, the shipping path.
    The long-form check needs the CT2 path for both, so it is refused with the
    hf backend. Everything that can be wrong with the arguments is checked
    before the first model loads. `predictions`, when a list, receives every
    utterance's reference and hypothesis for both models (tagged "candidate" /
    "baseline"); they are never part of the gate.
    """
    if not 0 <= min_improvement < 1:
        raise ValueError(f"min_improvement must be in [0, 1), got {min_improvement}")
    path = gate_path(candidate, out)
    path.parent.mkdir(parents=True, exist_ok=True)

    with_longform = longform_audio is not None or longform_reference is not None
    if with_longform:
        if longform_audio is None or longform_reference is None:
            raise ValueError("The long-form check needs both an audio file and a reference text")
        if backend != "ct2":
            raise ValueError("The long-form check runs the shipping CT2 path; export the model and use --backend ct2")
        if not Path(longform_audio).is_file():
            raise FileNotFoundError(f"Long-form audio not found: {longform_audio}")
        read_reference(longform_reference)  # unreadable or empty: fail now, not after the test split

    candidate_rows = [] if predictions is not None else None
    baseline_rows = [] if predictions is not None else None
    options = dict(device=device, compute_type=compute_type, language=language)
    candidate_result = evaluate_model(
        candidate, dataset_dir, split, backend=backend, predictions=candidate_rows, **options
    )
    baseline_result = evaluate_model(baseline, dataset_dir, split, backend="ct2", predictions=baseline_rows, **options)
    if predictions is not None:
        predictions.extend({"model": "candidate", **row} for row in candidate_rows)
        predictions.extend({"model": "baseline", **row} for row in baseline_rows)

    longform = None
    if with_longform:
        with tempfile.TemporaryDirectory(prefix="auditor-longform-") as workdir:
            longform = longform_check(
                candidate, baseline, longform_audio, longform_reference,
                tolerance=longform_tolerance, chunk_seconds=longform_chunk_seconds, workdir=workdir, **options,
            )

    checks = [
        _beats_baseline_check(candidate_result, baseline_result, min_improvement, split),
        _longform_check_entry(longform, require_longform),
    ]
    stored = ("split", "dataset_version")
    gate = {
        "schema": GATE_SCHEMA,
        "passed": all(c["passed"] for c in checks if c["passed"] is not None),
        "created_at": _utc_now(),
        "candidate": str(Path(candidate).resolve()) if Path(candidate).exists() else str(candidate),
        "baseline": str(baseline),
        "dataset_version": candidate_result["dataset_version"],
        "split": split,
        "checks": checks,
        "metrics": {
            "candidate": {k: v for k, v in candidate_result.items() if k not in stored},
            "baseline": {k: v for k, v in baseline_result.items() if k not in stored},
            "relative_wer_improvement": _relative_improvement(candidate_result, baseline_result),
        },
        "longform": longform,
    }
    _write_json(path, gate)
    logger.info("Gate %s written to %s", "passed" if gate["passed"] else "failed", path)
    return gate


# --- presentation (numbers and verdicts only) ---------------------------------------------


def _number(value):
    return "n/a" if value is None else f"{value:.4f}"


def _spread_line(label, spread):
    if spread is None:
        return f"{label}: n/a"
    return (
        f"{label} (n={spread['n']}): min {spread['min']:.4f}, median {spread['median']:.4f}, "
        f"p90 {spread['p90']:.4f}, max {spread['max']:.4f}"
    )


def render_gate(gate):
    """The gate as printable lines."""
    candidate, baseline = gate["metrics"]["candidate"], gate["metrics"]["baseline"]
    improvement = gate["metrics"]["relative_wer_improvement"]
    lines = [
        f"Dataset version {gate['dataset_version'] or 'unknown'}, split {gate['split']}: "
        f"{candidate['n']} utterances, {candidate['audio_seconds'] / 60:.1f} min of audio",
        f"{'':<10}{'WER norm':>10}{'CER norm':>10}{'WER raw':>10}{'CER raw':>10}",
    ]
    for label, metrics in (("candidate", candidate), ("baseline", baseline)):
        lines.append(
            f"{label:<10}{_number(metrics['wer_normalised']):>10}{_number(metrics['cer_normalised']):>10}"
            f"{_number(metrics['wer_raw']):>10}{_number(metrics['cer_raw']):>10}"
        )
    lines.append(
        "Relative WER improvement (normalised): "
        + ("n/a" if improvement is None else f"{improvement:+.1%}")
    )

    speakers = candidate.get("per_speaker")
    if speakers and speakers["n_speakers"]:
        lines.append(
            f"Candidate per-speaker WER: {speakers['n_speakers']} speakers, "
            f"{speakers['n_low_audio']} with under {speakers['min_audio_seconds']:g} s of audio"
        )
        lines.append("  " + _spread_line("all speakers", speakers["wer"]))
        lines.append("  " + _spread_line("enough audio", speakers["wer_reliable"]))

    longform = gate["longform"]
    if longform:
        lines.append(
            f"Long-form ({longform['audio_seconds'] / 60:.1f} min, {longform['reference_words']} reference words): "
            f"WER {_number(longform['candidate']['wer'])} vs baseline {_number(longform['baseline']['wer'])}"
        )
        for label in ("candidate", "baseline"):
            sanity = longform[label]["sanity"]
            lines.append(
                f"  {label}: {sanity['n_segments']} segments ({sanity['segments_per_minute']:.1f}/min), "
                f"{sanity['order_violations']} out of order, {sanity['long_segment_fraction']:.1%} window-long, "
                f"coverage {sanity['coverage']:.1%}"
            )

    marks = {True: "PASS", False: "FAIL", None: "SKIP"}
    lines.append("Checks:")
    lines.extend(f"  [{marks[c['passed']]}] {c['name']}: {c['detail']}" for c in gate["checks"])
    lines.append(f"Gate: {'PASSED' if gate['passed'] else 'FAILED'}")
    return lines
