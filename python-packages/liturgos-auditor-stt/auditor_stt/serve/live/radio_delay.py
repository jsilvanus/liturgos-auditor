"""Bounded adaptive radio-delay policy for caption/audio synchronisation.

This controller computes a requested delay only. The radio relay owns the audio
buffer and must apply the returned target without dropping or restarting audio.
"""
from dataclasses import dataclass


@dataclass
class AdaptiveDelay:
    minimum_seconds: float = 0.0
    maximum_seconds: float = 120.0
    initial_step_seconds: float = 0.5
    stable_reports_to_decrease: int = 20
    decrease_step_seconds: float = 0.25

    def __post_init__(self):
        if self.minimum_seconds < 0 or self.maximum_seconds < self.minimum_seconds:
            raise ValueError("invalid delay bounds")
        if self.initial_step_seconds <= 0 or self.decrease_step_seconds <= 0:
            raise ValueError("delay steps must be positive")
        if self.stable_reports_to_decrease < 1:
            raise ValueError("stable_reports_to_decrease must be positive")
        self.target_seconds = self.minimum_seconds
        self._stable_reports = 0
        self.last_reason = "initial"

    def report(self, mismatch_ms: float, tolerance_ms: float = 250.0):
        """Update target from measured audio/caption skew; positive means caption is late."""
        if tolerance_ms < 0:
            raise ValueError("tolerance_ms must be non-negative")
        if abs(mismatch_ms) <= tolerance_ms:
            self._stable_reports += 1
            if self._stable_reports >= self.stable_reports_to_decrease:
                self.target_seconds = max(
                    self.minimum_seconds,
                    round(self.target_seconds - self.decrease_step_seconds, 3),
                )
                self._stable_reports = 0
                self.last_reason = "stable_step_down"
            else:
                self.last_reason = "within_tolerance"
            return self.snapshot()

        self._stable_reports = 0
        # A caption that is already late needs more buffer. An early caption does
        # not justify increasing delay; report it for observability only.
        if mismatch_ms > tolerance_ms:
            needed = mismatch_ms / 1000.0
            step = max(self.initial_step_seconds, self.target_seconds - self.minimum_seconds + self.initial_step_seconds)
            increment = max(self.initial_step_seconds, min(needed, step))
            self.target_seconds = min(self.maximum_seconds, round(self.target_seconds + increment, 3))
            self.last_reason = "caption_late"
        else:
            self.last_reason = "caption_early"
        return self.snapshot()

    def request(self, seconds: float):
        """Explicit bounded request; useful for operator or relay control calls."""
        if seconds < self.minimum_seconds or seconds > self.maximum_seconds:
            raise ValueError(f"delay must be between {self.minimum_seconds:g} and {self.maximum_seconds:g} seconds")
        self.target_seconds = round(float(seconds), 3)
        self._stable_reports = 0
        self.last_reason = "explicit_request"
        return self.snapshot()

    def snapshot(self):
        return {
            "target_delay_seconds": self.target_seconds,
            "minimum_delay_seconds": self.minimum_seconds,
            "maximum_delay_seconds": self.maximum_seconds,
            "stable_reports": self._stable_reports,
            "stable_reports_to_decrease": self.stable_reports_to_decrease,
            "reason": self.last_reason,
        }
