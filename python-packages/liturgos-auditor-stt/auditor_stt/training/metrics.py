"""WER and CER without third-party dependencies (no jiwer, no torch).

Evaluation must run wherever the serving stack does, and the numbers must be
the same everywhere, so the edit distance is implemented here. It is Myers'
bit-vector algorithm on Python integers: exact Levenshtein distance, but fast
enough for the 10k-word sermon transcripts of the long-form check, where a
plain dynamic programme would take minutes.

Two views of every metric are reported. RAW is what the model printed against
the label as written: it punishes case and punctuation, which is what a
reader sees. NORMALISED lowercases and drops punctuation, so it measures
whether the right words were recognised; the evaluation gate uses that one.
"""

import unicodedata
from collections import defaultdict
from statistics import median

# Below this much audio a speaker's WER is a handful of words and mostly noise.
DEFAULT_MIN_SPEAKER_SECONDS = 30.0

_DIGITS = 6


def normalize_for_wer(text):
    """NFC, lowercase, punctuation and symbols to spaces, whitespace collapsed; digits are kept.

    Hyphens and dashes become spaces rather than being deleted: Whisper writes
    "kirkko-orkesteri" and "kirkko orkesteri" interchangeably, and deleting the
    hyphen would fuse the words ("kirkkoorkesteri") while a pause dash between
    words would glue them together. Digits stay because writing "15" for
    "viisitoista" is a genuine transcription difference, not formatting noise.
    """
    text = unicodedata.normalize("NFC", text or "").lower()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return " ".join(text.split())


def edit_distance(a, b):
    """Levenshtein distance (insert, delete and substitute all cost 1) between two sequences.

    Works on strings (characters) and on lists of hashable tokens (words).
    """
    if len(a) < len(b):
        a, b = b, a  # the bit vectors span the longer sequence; the loop runs over the shorter
    m = len(a)
    if not b:
        return m
    peq = {}  # symbol -> bit mask of the positions in `a` holding it
    for i, symbol in enumerate(a):
        peq[symbol] = peq.get(symbol, 0) | (1 << i)
    mask = (1 << m) - 1
    last = 1 << (m - 1)
    pv, mv, score = mask, 0, m  # vertical +1 / -1 deltas of the current column, and D[m][j]
    for symbol in b:
        eq = peq.get(symbol, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = (ph << 1) | 1  # a global distance: the top row of the table counts up by one
        mh <<= 1
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv & mask
    return score


def _rate(edits, reference_length):
    """Corpus-level error rate; None when there is no reference to measure against."""
    if reference_length == 0:
        return None
    return round(edits / reference_length, _DIGITS)


def _corpus_edits(references, hypotheses):
    """(sum of edit distances, sum of reference lengths) over aligned sequences.

    A pair with an empty reference adds its hypothesis length to the edits
    (every hypothesis token is an insertion) and nothing to the length.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    edits = sum(edit_distance(r, h) for r, h in zip(references, hypotheses))
    return edits, sum(len(r) for r in references)


def wer(references, hypotheses):
    """Corpus WER over whitespace-separated words: total edits / total reference words.

    Texts are used as given; normalise them first for a normalised WER. None
    when the references contain no words at all.
    """
    return _rate(*_corpus_edits([r.split() for r in references], [h.split() for h in hypotheses]))


def cer(references, hypotheses):
    """Corpus CER over characters (single spaces count as characters, as in jiwer).

    Whitespace runs are collapsed and the ends stripped first. None when the
    references contain no characters at all.
    """
    return _rate(*_corpus_edits([" ".join(r.split()) for r in references], [" ".join(h.split()) for h in hypotheses]))


def metrics_for(pairs):
    """WER and CER, raw and normalised, over (reference, hypothesis) pairs."""
    pairs = [(r or "", h or "") for r, h in pairs]
    raw_refs, raw_hyps = [r for r, _ in pairs], [h for _, h in pairs]
    refs, hyps = [normalize_for_wer(r) for r in raw_refs], [normalize_for_wer(h) for h in raw_hyps]
    return {
        "n": len(pairs),
        "reference_words": sum(len(r.split()) for r in refs),
        "wer_normalised": wer(refs, hyps),
        "cer_normalised": cer(refs, hyps),
        "wer_raw": wer(raw_refs, raw_hyps),
        "cer_raw": cer(raw_refs, raw_hyps),
    }


def _percentile(sorted_values, fraction):
    """Linear interpolation between closest ranks (numpy's default)."""
    position = (len(sorted_values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _spread(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], _DIGITS),
        "median": round(median(ordered), _DIGITS),
        "p90": round(_percentile(ordered, 0.9), _DIGITS),
        "max": round(ordered[-1], _DIGITS),
    }


def per_speaker(records, *, min_audio_seconds=DEFAULT_MIN_SPEAKER_SECONDS):
    """Normalised WER per speaker, and how it is spread across speakers.

    A pooled WER can look fine while one speaker is badly served, so the worst
    speakers stay visible. `records` are dicts with `speaker_id`, `reference`,
    `hypothesis` and `audio_seconds`; those without a speaker id are only
    counted. Speakers with less than `min_audio_seconds` of audio are flagged
    `low_audio`: `wer` spreads over everyone, `wer_reliable` over the rest
    (None when nobody qualifies).

    The result includes the per-speaker table under `speakers`; it holds
    pseudonymous ids, so drop it from anything that is stored.
    """
    grouped = defaultdict(list)
    unattributed = 0
    for record in records:
        if record.get("speaker_id"):
            grouped[record["speaker_id"]].append(record)
        else:
            unattributed += 1

    speakers = {}
    for speaker, items in grouped.items():
        refs = [normalize_for_wer(r["reference"]) for r in items]
        hyps = [normalize_for_wer(r["hypothesis"]) for r in items]
        seconds = sum(r.get("audio_seconds") or 0.0 for r in items)
        speakers[speaker] = {
            "wer": wer(refs, hyps),
            "utterances": len(items),
            "reference_words": sum(len(r.split()) for r in refs),
            "audio_seconds": round(seconds, 3),
            "low_audio": seconds < min_audio_seconds,
        }

    measured = [s for s in speakers.values() if s["wer"] is not None]
    return {
        "n_speakers": len(speakers),
        "n_low_audio": sum(1 for s in speakers.values() if s["low_audio"]),
        "unattributed_utterances": unattributed,
        "min_audio_seconds": min_audio_seconds,
        "wer": _spread([s["wer"] for s in measured]),
        "wer_reliable": _spread([s["wer"] for s in measured if not s["low_audio"]]),
        "speakers": speakers,
    }
