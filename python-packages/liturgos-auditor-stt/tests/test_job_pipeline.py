import shutil
import subprocess

import numpy as np
import pytest

from auditor_stt.serve.audio import MediaDecodeError
from auditor_stt.serve.jobs.chunking import Chunk
from auditor_stt.serve.jobs.pipeline import carry_prompt, transcribe_chunk, transcribe_file
from auditor_stt.serve.jobs.wav import open_pcm16_mono, write_pcm16_wav

RATE = 16000


class _Host:
    """Stub model: one segment per chunk whose text names the chunk's absolute start time."""

    loaded = True

    def __init__(self):
        self.calls = []

    def transcribe_array(
        self,
        samples,
        language=None,
        *,
        prompt=None,
        vad=False,
        temperature=None,
        condition_on_previous_text=None,
        word_timestamps=True,
        time_offset=0.0,
    ):
        self.calls.append(
            {
                "samples": len(samples),
                "dtype": samples.dtype,
                "language": language,
                "prompt": prompt,
                "vad": vad,
                "condition_on_previous_text": condition_on_previous_text,
                "word_timestamps": word_timestamps,
                "time_offset": time_offset,
            }
        )
        text = f"chunk at {time_offset:g}"
        end = time_offset + len(samples) / RATE
        word = {"start": time_offset, "end": end, "text": text, "probability": 0.9}
        segment = {"start": time_offset, "end": end, "text": text, "avg_logprob": -0.1, "no_speech_prob": 0.01}
        return {"text": text, "language": "fi", "segments": [{**segment, "words": [word]}]}


def _wav(tmp_path, seconds, name="source.wav"):
    path = tmp_path / name
    write_pcm16_wav(path, np.zeros(int(seconds * RATE), dtype=np.float32), RATE)
    return path


def _copy_normalize(source, wav_path):
    shutil.copyfile(source, wav_path)


def _no_gap(_samples, _rate):
    return None  # hard cuts on the nominal grid: chunk boundaries are predictable


# --- carry_prompt ----------------------------------------------------------------


def test_carry_prompt_is_the_user_prompt_plus_the_tail_of_the_previous_text():
    assert carry_prompt("kirkko", "edellinen lause", 200) == "kirkko edellinen lause"
    assert carry_prompt(None, "edellinen lause", 200) == "edellinen lause"
    assert carry_prompt("kirkko", None, 200) == "kirkko"
    assert carry_prompt("kirkko", "", 200) == "kirkko"
    assert carry_prompt(None, None, 200) is None
    assert carry_prompt("  ", "   ", 200) is None


def test_carry_prompt_cuts_the_tail_on_a_word_boundary():
    text = "alpha beta gamma delta"  # 22 chars
    assert carry_prompt(None, text, 12) == "gamma delta"  # the cut lands on the space before "gamma"
    assert carry_prompt(None, text, 13) == "gamma delta"  # would start at "a gamma": the "a" is a fragment
    assert carry_prompt(None, text, 11) == "gamma delta"  # exact fit
    assert carry_prompt(None, text, 10) == "delta"
    assert carry_prompt(None, text, 5) == "delta"
    assert carry_prompt(None, text, 4) is None  # only a fragment of "delta" would fit


def test_carry_prompt_never_exceeds_the_limit():
    text = " ".join(f"sana{i}" for i in range(200))
    for limit in (1, 7, 50, 200, 500):
        tail = carry_prompt(None, text, limit)
        assert tail is None or (len(tail) <= limit and text.endswith(tail))


def test_carry_prompt_keeps_the_user_prompt_whole_and_zero_disables_the_carry():
    long_text = "yksi kaksi kolme neljä viisi kuusi"
    assert carry_prompt("Jumalanpalvelus", long_text, 0) == "Jumalanpalvelus"
    assert carry_prompt(None, long_text, 0) is None
    assert carry_prompt("Jumalanpalvelus", long_text, 10) == "Jumalanpalvelus kuusi"


def test_carry_prompt_handles_a_tail_that_starts_at_whitespace():
    assert carry_prompt(None, "ab cd", 3) == "cd"  # tail " cd": its boundary is intact
    assert carry_prompt(None, "ab\ncd", 3) == "cd"


# --- transcribe_chunk ------------------------------------------------------------


def test_transcribe_chunk_reads_only_its_slice_and_reports_absolute_times(tmp_path):
    path = _wav(tmp_path, 30)
    host = _Host()
    chunk = Chunk(index=2, start_sample=10 * RATE, end_sample=17 * RATE, snapped=True)

    with open_pcm16_mono(path) as wav:
        result = transcribe_chunk(
            host,
            wav,
            chunk,
            language="fi",
            word_timestamps=False,
            prompt="kirkko",
            vad=True,
            previous_text="edellinen",
            carry_chars=200,
        )

    (call,) = host.calls
    assert call["samples"] == 7 * RATE  # this chunk's samples only, not the file
    assert call["dtype"] == np.float32
    assert call["time_offset"] == 10.0
    assert call["condition_on_previous_text"] is False
    assert call["word_timestamps"] is False
    assert call["vad"] is True
    assert call["language"] == "fi"
    assert call["prompt"] == "kirkko edellinen"

    assert set(result) == {"text", "language", "segments"}
    assert result["text"] == "chunk at 10"
    assert result["segments"][0]["start"] == 10.0
    assert result["segments"][0]["end"] == 17.0
    assert result["segments"][0]["words"][0]["start"] == 10.0


def test_transcribe_chunk_without_carry_over_sends_only_the_user_prompt(tmp_path):
    path = _wav(tmp_path, 10)
    host = _Host()
    chunk = Chunk(index=1, start_sample=5 * RATE, end_sample=10 * RATE, snapped=True)
    with open_pcm16_mono(path) as wav:
        transcribe_chunk(
            host, wav, chunk, language="fi", word_timestamps=True, prompt=None, vad=True,
            previous_text="edellinen", carry_chars=0,
        )  # fmt: skip
    assert host.calls[0]["prompt"] is None


# --- transcribe_file -------------------------------------------------------------


def test_transcribe_file_runs_the_whole_pipeline_in_absolute_time(tmp_path):
    source = _wav(tmp_path, 11)
    workdir = tmp_path / "work"
    workdir.mkdir()
    host, progress = _Host(), []

    result = transcribe_file(
        host,
        source,
        workdir=workdir,
        chunk_seconds=5,
        prompt="kirkko",
        normalize=_copy_normalize,
        find_gap=_no_gap,
        on_chunk=lambda index, total, chunk_result: progress.append((index, total, chunk_result["text"])),
    )

    assert [call["time_offset"] for call in host.calls] == [0.0, 5.0]  # 11 s: a 1 s tail joins the last chunk
    assert [call["samples"] for call in host.calls] == [5 * RATE, 6 * RATE]
    assert host.calls[0]["prompt"] == "kirkko"
    assert host.calls[1]["prompt"] == "kirkko chunk at 0"  # the previous chunk's text is carried over
    assert progress == [(0, 2, "chunk at 0"), (1, 2, "chunk at 5")]

    assert result["text"] == "chunk at 0 chunk at 5"
    assert result["language"] == "fi"
    assert result["complete"] is True
    assert (result["chunks_done"], result["chunks_total"]) == (2, 2)
    assert result["duration_seconds"] == pytest.approx(11.0)
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(0.0, 5.0), (5.0, 11.0)]


def test_transcribe_file_cleans_up_and_never_touches_the_source(tmp_path):
    source = _wav(tmp_path, 6)
    workdir = tmp_path / "work"
    workdir.mkdir()

    transcribe_file(_Host(), source, workdir=workdir, chunk_seconds=5, normalize=_copy_normalize, find_gap=_no_gap)

    assert list(workdir.iterdir()) == []
    assert source.exists()


def test_transcribe_file_cleans_up_when_the_model_fails(tmp_path):
    class _Broken(_Host):
        def transcribe_array(self, samples, language=None, **options):
            raise RuntimeError("model exploded")

    source = _wav(tmp_path, 6)
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(RuntimeError, match="model exploded"):
        transcribe_file(_Broken(), source, workdir=workdir, normalize=_copy_normalize, find_gap=_no_gap)

    assert list(workdir.iterdir()) == []
    assert source.exists()


def test_transcribe_file_rejects_media_without_audio(tmp_path):
    source = tmp_path / "empty.wav"
    write_pcm16_wav(source, np.zeros(0, dtype=np.float32), RATE)
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(MediaDecodeError, match="no audio"):
        transcribe_file(_Host(), source, workdir=workdir, normalize=_copy_normalize, find_gap=_no_gap)
    assert list(workdir.iterdir()) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_transcribe_file_normalises_a_real_container(tmp_path):
    source = tmp_path / "clip.m4a"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=8"]
        + ["-ar", "44100", "-ac", "2", str(source)],
        check=True,
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    host = _Host()

    result = transcribe_file(host, source, workdir=workdir, chunk_seconds=5, find_gap=_no_gap)

    assert host.calls[0]["samples"] == 5 * RATE  # the 16 kHz mono conversion happened
    assert result["complete"] is True
    assert result["duration_seconds"] == pytest.approx(8.0, abs=0.15)
    assert list(workdir.iterdir()) == []
