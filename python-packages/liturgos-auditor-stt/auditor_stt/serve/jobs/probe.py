"""Audio duration of a job's source, read with ffprobe at submit.

A queued job has no normalised audio yet, so without this the service cannot say how
much audio waits ahead of a job. The probe is best effort: a missing ffprobe, a file
ffprobe cannot read or a timeout all give None and the job's duration stays unknown
until the runner has normalised it.
"""

import math
import subprocess


class ProbeError(Exception):
    """ffprobe could not tell the duration (the message is for logs only)."""


def probe_local(path, timeout):
    """Duration in seconds of a local media file."""
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise ProbeError(f"ffprobe did not finish in {timeout:g} s") from None
    except OSError as exc:
        raise ProbeError(f"ffprobe could not be started ({type(exc).__name__})") from None
    if result.returncode != 0:
        raise ProbeError("ffprobe could not read the file")
    return parse_duration(result.stdout.strip())


def parse_duration(value):
    """A positive, finite number of seconds from ffprobe's text or JSON value."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ProbeError("ffprobe reported no duration") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ProbeError("ffprobe reported no usable duration")
    return seconds
