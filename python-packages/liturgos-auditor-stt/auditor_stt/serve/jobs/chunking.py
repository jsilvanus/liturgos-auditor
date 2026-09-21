"""Chunk planning for batch jobs.

The file is cut on a nominal grid (k * chunk_seconds). Each internal cut is
moved into the middle of the longest non-speech gap within +/- snap_window so
words are rarely split; when there is no usable gap the cut stays on the grid
and is flagged. Chunks do not overlap, so results concatenate without dedupe.

Only the samples around each boundary are read, so planning a 12 hour file
touches a few percent of it. The plan is persisted in the job manifest by the
caller and never recomputed on resume: a different VAD result after a restart
must not shift already-transcribed chunks.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # the normalised rate; plan_chunks takes the real one from the WAV

MIN_CHUNK_SECONDS = 5.0
MAX_CHUNK_SECONDS = 300.0
MIN_TAIL_FRACTION = 0.25  # a shorter final piece is merged into the previous chunk
MIN_GAP_SECONDS = 0.3
RMS_FRAME_SECONDS = 0.1


@dataclass(frozen=True)
class Chunk:
    """Samples [start_sample, end_sample) of the job audio.

    `snapped` describes the cut that ENDS this chunk: True when it sits in a
    detected non-speech gap (or is the end of the file), False for a hard cut
    on the nominal grid, i.e. the next chunk starts mid-speech.
    """

    index: int
    start_sample: int
    end_sample: int
    snapped: bool
    sample_rate: int = SAMPLE_RATE

    @property
    def start(self):
        return self.start_sample / self.sample_rate

    @property
    def end(self):
        return self.end_sample / self.sample_rate

    def to_dict(self):
        """The manifest entry."""
        return {
            "index": self.index,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start": self.start,
            "end": self.end,
            "snapped": self.snapped,
        }

    @classmethod
    def from_dict(cls, data, sample_rate):
        return cls(data["index"], data["start_sample"], data["end_sample"], data["snapped"], sample_rate)


def plan_chunks(wav, chunk_seconds=60.0, snap_window=5.0, find_gap=None):
    """Plan the chunks of `wav` (anything with sample_rate, num_samples and read()).

    `find_gap(window_samples, sample_rate)` returns the index of the cut inside
    the window, or None; the default looks for silence with Silero VAD. An
    empty file has no chunks. A final piece shorter than a quarter of
    `chunk_seconds` is merged into the previous chunk (judged on the nominal
    grid, before snapping).
    """
    if not MIN_CHUNK_SECONDS <= chunk_seconds <= MAX_CHUNK_SECONDS:
        raise ValueError(f"chunk_seconds must be between {MIN_CHUNK_SECONDS:g} and {MAX_CHUNK_SECONDS:g}")
    # Windows of neighbouring boundaries must not overlap, or cuts could cross.
    if not 0 <= snap_window < chunk_seconds / 2:
        raise ValueError("snap_window must be at least 0 and less than half of chunk_seconds")

    find_gap = find_gap or default_find_gap
    rate = wav.sample_rate
    total = wav.num_samples
    if total == 0:
        return []
    nominal = round(chunk_seconds * rate)
    window = min(round(snap_window * rate), (nominal - 1) // 2)

    boundaries = list(range(nominal, total, nominal))
    if boundaries and total - boundaries[-1] < MIN_TAIL_FRACTION * nominal:
        boundaries.pop()

    cuts = [_snap(wav, boundary, window, find_gap) for boundary in boundaries]
    starts = [0] + [cut for cut, _ in cuts]
    ends = [cut for cut, _ in cuts] + [total]
    flags = [snapped for _, snapped in cuts] + [True]
    return [Chunk(i, start, end, snapped, rate) for i, (start, end, snapped) in enumerate(zip(starts, ends, flags))]


def _snap(wav, boundary, window, find_gap):
    if window == 0:
        return boundary, False
    low = max(0, boundary - window)
    high = min(wav.num_samples, boundary + window)
    offset = find_gap(wav.read(low, high), wav.sample_rate)
    # Anything outside the open window would risk an empty or reordered chunk.
    if offset is not None and 0 < offset < high - low:
        return low + int(offset), True
    return boundary, False


def default_find_gap(samples, sample_rate):
    try:
        return _vad_gap(samples, sample_rate)
    except Exception as exc:  # noqa: BLE001 - a broken VAD must not fail the job; RMS still finds a quiet spot
        logger.warning("VAD failed (%s); using the quietest 100 ms frame instead", type(exc).__name__)
        return _quietest_frame(samples, sample_rate)


def _vad_gap(samples, sample_rate):
    """Middle of the longest gap between speech segments, or None if all gaps are short."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    if sample_rate != SAMPLE_RATE:
        raise ValueError("Silero VAD needs 16 kHz audio")
    if len(samples) == 0:
        return None

    # Report pauses as short as MIN_GAP_SECONDS (the default would bridge anything under 2 s).
    options = VadOptions(min_silence_duration_ms=int(MIN_GAP_SECONDS * 1000), speech_pad_ms=30)
    speech = get_speech_timestamps(samples, options, sampling_rate=sample_rate)

    gaps = []
    cursor = 0
    for segment in speech:
        gaps.append((cursor, segment["start"]))
        cursor = segment["end"]
    gaps.append((cursor, len(samples)))

    min_gap = MIN_GAP_SECONDS * sample_rate
    centre = len(samples) // 2
    usable = [(start, end) for start, end in gaps if end - start >= min_gap]
    if not usable:
        return None
    # Longest gap wins; on a tie prefer the one nearest the nominal boundary.
    start, end = max(usable, key=lambda gap: (gap[1] - gap[0], -abs((gap[0] + gap[1]) // 2 - centre)))
    return (start + end) // 2


def _quietest_frame(samples, sample_rate):
    """Centre of the lowest-RMS 100 ms frame, nearest the window centre among equals."""
    frame = int(RMS_FRAME_SECONDS * sample_rate)
    count = len(samples) // frame if frame else 0
    if count < 1:
        return None
    frames = samples[: count * frame].astype(np.float64).reshape(count, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    quietest = np.flatnonzero(rms <= rms.min() + 1e-9)
    best = quietest[np.argmin(np.abs(quietest - (count - 1) / 2))]
    return int(best) * frame + frame // 2
