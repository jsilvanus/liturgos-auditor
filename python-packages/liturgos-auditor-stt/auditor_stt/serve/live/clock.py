"""Wall-clock time of a sample.

The PCM of a live stream carries no clock, so the service pins one: the wall time
at which the first sample arrived. Every later sample's time is that anchor plus
its position, which stays right as long as no samples are dropped. If the source
skips (a network gap, a restarted ffmpeg) the position runs behind the wall; the
reader notices while it is keeping up with the stream and moves the anchor.
"""

import time


class WallClock:
    def __init__(self, sample_rate=16000, *, now=time.time, drift_limit=1.0):
        self.sample_rate = sample_rate
        self.drift_limit = drift_limit
        self._now = now
        self._anchor_wall = None
        self._anchor_sample = 0

    @property
    def anchored(self):
        return self._anchor_wall is not None

    def anchor(self, sample, wall=None):
        self._anchor_sample = sample
        self._anchor_wall = self._now() if wall is None else wall

    def time_of(self, sample):
        if self._anchor_wall is None:
            raise RuntimeError("the clock has no anchor yet")
        return self._anchor_wall + (sample - self._anchor_sample) / self.sample_rate

    def check(self, received_samples, *, caught_up):
        """Called after each read; `received_samples` counts everything read so far.

        Only a reader that has no backlog can compare the stream position with the
        wall (a backlog is old audio and the anchor is still right for it). Returns
        the drift in seconds when the anchor was moved, else None.
        """
        if not self.anchored:
            self.anchor(received_samples)
            return None
        if not caught_up:
            return None
        arrival = self._now()
        drift = arrival - self.time_of(received_samples)
        if abs(drift) > self.drift_limit:
            self.anchor(received_samples, arrival)
            return drift
        return None


def iso(wall):
    """UTC with milliseconds, e.g. 2026-10-05T09:30:00.250Z."""
    whole = int(wall)
    millis = int(round((wall - whole) * 1000))
    if millis == 1000:
        whole, millis = whole + 1, 0
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(whole)) + f".{millis:03d}Z"
