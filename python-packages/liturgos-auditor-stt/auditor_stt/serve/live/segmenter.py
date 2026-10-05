"""Cut a live PCM stream into speech segments at pauses.

Feed float32 mono samples, get back segments ready to transcribe. A segment ends
where speech has paused for `pause_seconds`, or inside the longest gap once it
reaches `max_seconds` (a hard cut only when it has no gap at all). Silence is not
returned. Sample positions are absolute (counted from the first fed sample), so
the caller can map them to wall time.
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Segment:
    start_sample: int
    end_sample: int
    samples: np.ndarray


def energy_speech_ranges(samples, sample_rate=SAMPLE_RATE):
    """(start, end) sample ranges that are louder than the background; the fallback when Silero is unavailable."""
    frame = int(0.03 * sample_rate)
    count = len(samples) // frame
    if count < 1:
        return []
    rms = np.sqrt(np.mean(samples[: count * frame].astype(np.float64).reshape(count, frame) ** 2, axis=1))
    # The background is estimated from the quietest tenth, but a buffer that is all speech must not make speech the background.
    threshold = max(0.008, 3.0 * min(float(np.percentile(rms, 10)), 0.01))
    voiced = rms > threshold
    hang = 5  # keep 150 ms of tail so that short dips do not split a word
    ranges, start, last = [], None, None
    for index, flag in enumerate(voiced):
        if flag:
            if start is None:
                start = index
            last = index
        elif start is not None and index - last > hang:
            ranges.append((start * frame, (last + 1) * frame))
            start = None
    if start is not None:
        ranges.append((start * frame, (last + 1) * frame))
    return [(s, e) for s, e in ranges if e - s >= 0.1 * sample_rate]


def vad_speech_ranges(samples, sample_rate=SAMPLE_RATE):
    """Speech ranges from Silero VAD (as shipped with faster-whisper), falling back to loudness if it fails."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        options = VadOptions(min_silence_duration_ms=200, speech_pad_ms=30)
        return [(t["start"], t["end"]) for t in get_speech_timestamps(samples, options, sampling_rate=sample_rate)]
    except Exception:  # noqa: BLE001 - a broken VAD must not end a live session
        return energy_speech_ranges(samples, sample_rate)


class Segmenter:
    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        *,
        min_seconds=1.0,
        max_seconds=10.0,
        pause_seconds=0.6,
        flush_pause_seconds=2.0,
        check_seconds=0.4,
        pad_seconds=0.15,
        speech_ranges=vad_speech_ranges,
    ):
        self.rate = sample_rate
        self.min = int(min_seconds * sample_rate)
        self.max = int(max_seconds * sample_rate)
        self.pause = int(pause_seconds * sample_rate)
        self.flush_pause = int(flush_pause_seconds * sample_rate)
        self.check = int(check_seconds * sample_rate)
        self.pad = int(pad_seconds * sample_rate)
        self._ranges = speech_ranges
        self._buffer = np.zeros(0, dtype=np.float32)
        self._start = 0  # absolute sample index of _buffer[0]
        self._fresh = 0  # samples fed since the last look

    def feed(self, samples):
        self._buffer = np.concatenate([self._buffer, np.asarray(samples, dtype=np.float32)])
        self._fresh += len(samples)
        if self._fresh < self.check and len(self._buffer) < self.max:
            return []
        self._fresh = 0
        out = []
        while True:
            segment = self._next()
            if segment is None:
                return out
            out.append(segment)

    def flush(self):
        """End of stream: whatever speech is left."""
        ranges = self._ranges(self._buffer, self.rate) if len(self._buffer) else []
        if not ranges:
            self._drop(len(self._buffer))
            return None
        lead = max(0, ranges[0][0] - self.pad)
        end = min(len(self._buffer), ranges[-1][1] + self.pad)
        return self._take(lead, end)

    # --- internals -------------------------------------------------------

    def _drop(self, count):
        self._buffer = self._buffer[count:]
        self._start += count

    def _take(self, lead, end):
        segment = Segment(self._start + lead, self._start + end, self._buffer[lead:end].copy())
        self._drop(end)
        return segment

    def _next(self):
        buffer = self._buffer
        if len(buffer) == 0:
            return None
        ranges = self._ranges(buffer, self.rate)
        if not ranges:
            self._drop(max(0, len(buffer) - 2 * self.pad))  # keep a little lead-in for the next word
            return None
        lead = max(0, ranges[0][0] - self.pad)
        if lead:
            self._drop(lead)
            ranges = [(s - lead, e - lead) for s, e in ranges]
            buffer = self._buffer
        last_end = ranges[-1][1]
        quiet = len(buffer) - last_end
        cut = min(len(buffer), last_end + self.pad)
        if cut <= self.max and (
            (quiet >= self.pause and cut >= self.min) or quiet >= self.flush_pause
        ):
            return self._take(0, cut)
        if len(buffer) >= self.max:
            return self._take(0, self._gap_cut(ranges))
        return None

    def _gap_cut(self, ranges):
        best, best_length = None, 0
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            if end >= self.max:
                break
            length = start - end
            if length > best_length:
                best, best_length = min((end + start) // 2, self.max), length
        return best if best is not None and best >= self.rate // 2 else self.max
