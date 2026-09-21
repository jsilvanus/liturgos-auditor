import shutil
import subprocess
import threading

import numpy as np
import pytest

from auditor_stt.serve import audio
from auditor_stt.serve.audio import MediaDecodeError, NormalizationCancelledError, normalize_to_wav
from auditor_stt.serve.jobs.wav import open_pcm16_mono, write_pcm16_wav

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def _sine(path, *, seconds=2, rate=44100, channels=2):
    """A test input in whatever container the extension of `path` selects."""
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate={rate}", "-ac", channels, path)
    return path


def _leftovers(directory):
    return sorted(p.name for p in directory.iterdir())


@needs_ffmpeg
@pytest.mark.parametrize("name", ["tone.wav", "tone.m4a", "tone.mp3", "tone.ogg"])
def test_any_container_becomes_16khz_mono_pcm16(tmp_path, name):
    source = _sine(tmp_path / name)
    target = tmp_path / "out" / "audio.wav"
    target.parent.mkdir()

    normalize_to_wav(source, target)

    with open_pcm16_mono(target) as wav:  # raises unless it is 16-bit PCM mono
        assert wav.sample_rate == 16000
        assert wav.duration == pytest.approx(2.0, abs=0.15)
        samples = wav.read(0, wav.num_samples)
    assert np.abs(samples).max() > 0.05  # the tone survived (lavfi's sine peaks near 0.125); not silence
    assert _leftovers(target.parent) == ["audio.wav"]  # no .partial file left behind
    assert source.exists()  # the input is never touched


@needs_ffmpeg
def test_video_container_keeps_only_the_audio(tmp_path):
    source = tmp_path / "clip.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=2",
        "-shortest", "-c:v", "libx264", "-c:a", "aac", source,
    )  # fmt: skip
    normalize_to_wav(source, tmp_path / "audio.wav")

    with open_pcm16_mono(tmp_path / "audio.wav") as wav:
        assert wav.sample_rate == 16000
        assert wav.duration == pytest.approx(2.0, abs=0.2)


@needs_ffmpeg
def test_an_existing_target_is_replaced(tmp_path):
    source = _sine(tmp_path / "tone.wav", seconds=1)
    target = tmp_path / "audio.wav"
    write_pcm16_wav(target, np.zeros(16000 * 5, dtype=np.float32), 16000)

    normalize_to_wav(source, target)

    with open_pcm16_mono(target) as wav:
        assert wav.duration == pytest.approx(1.0, abs=0.1)


@needs_ffmpeg
def test_undecodable_input_raises_without_leaking_the_server_path(tmp_path):
    directory = tmp_path / "secret-server-dir"
    directory.mkdir()
    source = directory / "garbage.mp4"
    source.write_bytes(b"not audio at all " * 200)
    target = directory / "audio.wav"

    with pytest.raises(MediaDecodeError) as excinfo:
        normalize_to_wav(source, target)

    message = str(excinfo.value)
    assert "garbage.mp4" in message
    assert "secret-server-dir" not in message
    assert str(tmp_path) not in message and str(tmp_path).replace("\\", "/") not in message
    assert _leftovers(directory) == ["garbage.mp4"]  # neither audio.wav nor a .partial file


@needs_ffmpeg
def test_input_without_an_audio_stream_is_a_decode_error(tmp_path):
    source = tmp_path / "silent-video.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=64x64:rate=5:duration=1", "-c:v", "libx264", source)

    with pytest.raises(MediaDecodeError):
        normalize_to_wav(source, tmp_path / "audio.wav")
    assert not (tmp_path / "audio.wav").exists()


@needs_ffmpeg
def test_missing_source_names_only_the_file(tmp_path):
    with pytest.raises(MediaDecodeError, match="Source file not found: gone.mp4") as excinfo:
        normalize_to_wav(tmp_path / "gone.mp4", tmp_path / "audio.wav")
    assert str(tmp_path) not in str(excinfo.value)


@needs_ffmpeg
def test_playlists_cannot_pull_in_other_protocols(tmp_path):
    # An uploaded .m3u8 naming a URL must not make the server fetch it (SSRF / local-file oracle).
    playlist = tmp_path / "evil.m3u8"
    playlist.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://127.0.0.1:9/x.ts\n#EXT-X-ENDLIST\n")

    with pytest.raises(MediaDecodeError, match="whitelist"):  # refused up front, not a failed connection
        normalize_to_wav(playlist, tmp_path / "audio.wav")


def test_missing_ffmpeg_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)
    source = tmp_path / "tone.wav"
    source.write_bytes(b"x")

    with pytest.raises(MediaDecodeError, match="ffmpeg is not installed"):
        normalize_to_wav(source, tmp_path / "audio.wav")


@needs_ffmpeg
def test_cancel_event_set_beforehand_never_starts_ffmpeg(tmp_path):
    source = _sine(tmp_path / "tone.wav", seconds=1)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(NormalizationCancelledError):
        normalize_to_wav(source, tmp_path / "audio.wav", cancel=cancel)
    assert _leftovers(tmp_path) == ["tone.wav"]


class _HangingProcess:
    """Stands in for a long ffmpeg run: communicate() times out until the process is killed."""

    returncode = None

    def __init__(self):
        self.killed = False

    def communicate(self, timeout=None):
        if self.killed:
            return b"", b""
        raise subprocess.TimeoutExpired("ffmpeg", timeout)

    def poll(self):
        return None if not self.killed else -9

    def kill(self):
        self.killed = True


@pytest.fixture
def hanging_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "long.mp4"
    source.write_bytes(b"x")
    process = _HangingProcess()
    monkeypatch.setattr(audio.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(audio, "_POLL_SECONDS", 0.01)
    return source, process


def test_cancel_event_kills_a_running_ffmpeg(tmp_path, hanging_ffmpeg):
    source, process = hanging_ffmpeg
    cancel = threading.Event()
    threading.Timer(0.05, cancel.set).start()

    with pytest.raises(NormalizationCancelledError):
        normalize_to_wav(source, tmp_path / "audio.wav", cancel=cancel)
    assert process.killed


def test_timeout_kills_ffmpeg_and_reports_it(tmp_path, hanging_ffmpeg):
    source, process = hanging_ffmpeg

    with pytest.raises(MediaDecodeError, match="timed out"):
        normalize_to_wav(source, tmp_path / "audio.wav", timeout=0.05)
    assert process.killed
