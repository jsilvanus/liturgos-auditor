"""Audio normalisation and content-addressed storage for synced recordings.

crowd-source-voice accepts WAV, WebM and Ogg at whatever sample rate the
browser recorded, so every recording is converted to 16 kHz mono PCM16 WAV
before it is hashed. The hash of the *normalised* bytes names the file
(`audio/<sha256>.wav`), which makes identical audio share one file no matter
what container it arrived in.
"""

import io
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

FFMPEG_TIMEOUT_SECONDS = 120


class AudioNormalizeError(Exception):
    """This recording could not be converted; other recordings are unaffected."""


class AudioNormalizeTimeout(AudioNormalizeError):
    """Conversion did not finish in time. Unlike a bad file this is transient (a busy machine), so it is retried."""


class FfmpegNotFoundError(RuntimeError):
    """ffmpeg is not installed, so no recording can be converted."""


def normalize_audio(data, source_name=None):
    """Convert audio bytes in any ffmpeg-readable container to 16 kHz mono PCM16 WAV bytes.

    Metadata and encoder tags are stripped (`bitexact`) so the same audio
    yields the same bytes, and therefore the same hash, across ffmpeg versions.
    The output goes to a file rather than a pipe: ffmpeg cannot seek a pipe to
    fill in the WAV header sizes, and the `wave` module then misreads the length.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FfmpegNotFoundError("ffmpeg was not found on PATH; it is needed to normalise recordings")

    suffix = Path(source_name).suffix if source_name else ""
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,5}", suffix):
        suffix = ""

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in" + suffix)
        out = os.path.join(tmp, "out.wav")
        with open(src, "wb") as f:
            f.write(data)
        try:
            proc = subprocess.run(
                [
                    ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", src, "-vn", "-map_metadata", "-1",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    "-fflags", "+bitexact", "-flags:a", "+bitexact", out,
                ],
                capture_output=True,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioNormalizeTimeout("ffmpeg timed out") from exc
        if proc.returncode != 0 or not os.path.exists(out):
            detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"]
            raise AudioNormalizeError(f"ffmpeg exited with {proc.returncode}: {detail[0][:200]}")
        with open(out, "rb") as f:
            return f.read()


def wav_duration(wav_bytes):
    """Length in seconds, read from the WAV header of already-normalised audio."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            rate = w.getframerate()
            if rate <= 0:
                raise AudioNormalizeError("WAV has no sample rate")
            return w.getnframes() / rate
    except (wave.Error, EOFError) as exc:
        raise AudioNormalizeError(f"not a readable WAV: {exc}") from exc


def audio_path(data_dir, sha256):
    return Path(data_dir) / "audio" / f"{sha256}.wav"


def store_audio(data_dir, sha256, wav_bytes):
    """Write `audio/<sha256>.wav` unless it already exists; the rename makes a crash leave no partial file."""
    target = audio_path(data_dir, sha256)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(wav_bytes)
    os.replace(tmp, target)
    return target
