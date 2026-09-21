"""Audio staging for the service: temp files for uploaded live segments, and
ffmpeg normalisation of whole media files for batch jobs.

faster-whisper decodes arbitrary containers (fMP4/AAC, WAV, ...) itself via
PyAV, so live segments only need their bytes written to a suffixed temp file
and cleaned up afterwards. Batch jobs instead decode the input ONCE with ffmpeg
into a 16 kHz mono WAV on disk (same flags as scripts/extract-audio.sh), which
the runner then slices chunk by chunk without ever holding the file in memory.
"""

import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

_POLL_SECONDS = 0.25
_STDERR_LIMIT = 300


class MediaDecodeError(Exception):
    """ffmpeg could not turn the input into audio (or is not installed)."""


class NormalizationCancelledError(Exception):
    """normalize_to_wav was stopped through its cancel event."""


def _suffix_for(filename, content_type):
    if filename and "." in filename:
        return os.path.splitext(filename)[1]
    if content_type and "wav" in content_type:
        return ".wav"
    return ".mp4"


@contextmanager
def temp_audio_file(data, filename=None, content_type=None):
    suffix = _suffix_for(filename, content_type)
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _scrub(text, *paths):
    """Replace server paths in ffmpeg's stderr by bare file names; it echoes the input path."""
    variants = set()
    for path in paths:
        for candidate in (os.fspath(path), os.path.abspath(path)):
            variants.update({candidate, candidate.replace("\\", "/")})
    for variant in sorted(variants, key=len, reverse=True):
        text = text.replace(variant, os.path.basename(variant))
    for path in paths:
        directory = os.path.dirname(os.path.abspath(path))
        for variant in {directory, directory.replace("\\", "/")}:
            text = text.replace(variant + os.sep, "").replace(variant + "/", "")
    return text


def _replace(source, target):
    # Windows refuses to replace a file another handle has open (an antivirus scan right
    # after ffmpeg closed it); that clears within moments.
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _remove(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def normalize_to_wav(source_path, wav_path, *, timeout=None, cancel=None):
    """Decode `source_path` (any container ffmpeg reads) to a 16 kHz mono PCM16 WAV at `wav_path`.

    Blocking: callers run it in a thread. ffmpeg writes to a temp name next to
    the target and the result is renamed into place, so `wav_path` either does
    not exist or is complete, even if the process dies half way through.

    `cancel` is an optional threading.Event; setting it kills ffmpeg and raises
    NormalizationCancelledError, so a 50 GB input does not have to run to the
    end before a shutdown or a DELETE takes effect. `timeout` is in seconds.
    Errors name the input file but never the server's directory layout.
    """
    source, target = Path(source_path), Path(wav_path)
    partial = target.with_name(target.name + ".partial")
    name = source.name

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MediaDecodeError("ffmpeg is not installed on the server")
    if not source.is_file():
        raise MediaDecodeError(f"Source file not found: {name}")
    if cancel is not None and cancel.is_set():
        raise NormalizationCancelledError()

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    # A playlist or container inside the input may name other URLs; only local files are ever read.
    command += ["-protocol_whitelist", "file", "-i", str(source)]
    command += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(partial)]
    deadline = None if timeout is None else time.monotonic() + timeout
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        while True:
            try:
                _, stderr = proc.communicate(timeout=_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    raise NormalizationCancelledError() from None
                if deadline is not None and time.monotonic() >= deadline:
                    raise MediaDecodeError(f"Decoding {name} timed out after {timeout:g} s") from None
        if proc.returncode != 0:
            lines = _scrub(stderr.decode("utf-8", "replace"), source, partial).splitlines()
            reason = " | ".join(line.strip() for line in lines if line.strip())[:_STDERR_LIMIT]
            reason = reason or f"ffmpeg exited with status {proc.returncode}"
            raise MediaDecodeError(f"Could not decode audio from {name}: {reason}")
        _replace(partial, target)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        _remove(partial)
        raise
