"""How fast finished jobs ran, for estimating how long a queue takes.

The rate is audio seconds per wall-clock second over the last few completed jobs
(median, so one odd job does not move it). It is measured end to end, so it includes
normalising and any time live captions held the model. Nothing is measured at startup:
until a job has finished the rate is None.
"""

import json
import logging
import os
import statistics
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

KEEP = 5


class RateTracker:
    def __init__(self, path, keep=KEEP):
        self._path = path
        self._keep = keep
        self._lock = threading.Lock()
        self._samples = self._load()

    def _load(self):
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return []
        samples = data.get("samples") if isinstance(data, dict) else None
        if not isinstance(samples, list):
            return []
        valid = [
            s for s in samples
            if isinstance(s, dict) and _positive(s.get("audio_seconds")) and _positive(s.get("wall_seconds"))
        ]
        return valid[-self._keep:]

    def record(self, audio_seconds, wall_seconds):
        """Remember one finished job; ignored when either number is not positive."""
        if not (_positive(audio_seconds) and _positive(wall_seconds)):
            return
        sample = {
            "audio_seconds": round(float(audio_seconds), 3),
            "wall_seconds": round(float(wall_seconds), 3),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            self._samples = (self._samples + [sample])[-self._keep:]
            tmp = f"{self._path}.tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"samples": self._samples}, f)
                os.replace(tmp, self._path)
            except OSError:
                logger.warning("Could not save the job rate samples")

    def rate(self):
        """Audio seconds per wall second (median of the kept samples), or None."""
        with self._lock:
            if not self._samples:
                return None
            return statistics.median(s["audio_seconds"] / s["wall_seconds"] for s in self._samples)

    def sample_count(self):
        with self._lock:
            return len(self._samples)


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
