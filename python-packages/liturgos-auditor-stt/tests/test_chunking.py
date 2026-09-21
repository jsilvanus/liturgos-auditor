import numpy as np
import pytest

from auditor_stt.serve.jobs import chunking
from auditor_stt.serve.jobs.chunking import Chunk, default_find_gap, plan_chunks
from auditor_stt.serve.jobs.wav import open_pcm16_mono, write_pcm16_wav


class FakeWav:
    """Just enough of Pcm16Wav for planning; records which samples were read."""

    def __init__(self, seconds, sample_rate=100):
        self.sample_rate = sample_rate
        self.num_samples = round(seconds * sample_rate)
        self.reads = []

    def read(self, start, end):
        self.reads.append((start, end))
        return np.zeros(end - start, dtype=np.float32)


def _never(_samples, _rate):
    return None


def _ranges(chunks):
    return [(c.start_sample, c.end_sample) for c in chunks]


def _assert_contiguous(chunks, total):
    assert chunks[0].start_sample == 0
    assert chunks[-1].end_sample == total
    for previous, following in zip(chunks, chunks[1:]):
        assert previous.end_sample == following.start_sample
    assert all(c.start_sample < c.end_sample for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


# --- grid, snapping and flags (injected finder) ----------------------------


def test_hard_cuts_stay_on_the_nominal_grid_and_are_flagged():
    chunks = plan_chunks(FakeWav(150), chunk_seconds=60, find_gap=_never)
    assert _ranges(chunks) == [(0, 6000), (6000, 12000), (12000, 15000)]
    # The flag describes the cut that ends the chunk; the file end is never a hard cut.
    assert [c.snapped for c in chunks] == [False, False, True]


def test_snapped_cut_uses_the_index_the_finder_returns():
    wav = FakeWav(150)
    calls = []

    def finder(samples, rate):
        calls.append((len(samples), rate, samples.dtype))
        return 300

    chunks = plan_chunks(wav, chunk_seconds=60, snap_window=5, find_gap=finder)

    # Window is [b - 500, b + 500) samples, so index 300 is 2 s before the boundary.
    assert _ranges(chunks) == [(0, 5800), (5800, 11800), (11800, 15000)]
    assert all(c.snapped for c in chunks)
    assert calls == [(1000, 100, np.float32)] * 2
    assert wav.reads == [(5500, 6500), (11500, 12500)]  # only the windows are read


def test_a_gap_less_boundary_is_hard_cut_while_others_snap():
    answers = iter([250, None])
    chunks = plan_chunks(FakeWav(150), 60, 5, find_gap=lambda samples, rate: next(answers))
    assert _ranges(chunks) == [(0, 5750), (5750, 12000), (12000, 15000)]
    assert [c.snapped for c in chunks] == [True, False, True]


@pytest.mark.parametrize("bad_index", [0, -1, 1000, 5000])
def test_out_of_window_finder_answers_are_treated_as_no_gap(bad_index):
    chunks = plan_chunks(FakeWav(100), 60, 5, find_gap=lambda samples, rate: bad_index)
    assert _ranges(chunks) == [(0, 6000), (6000, 10000)]
    assert chunks[0].snapped is False


def test_snap_window_zero_disables_snapping():
    calls = []
    chunks = plan_chunks(FakeWav(130), 60, 0, find_gap=lambda *a: calls.append(a))
    assert calls == []
    assert _ranges(chunks) == [(0, 6000), (6000, 13000)]


@pytest.mark.parametrize("pick", ["first", "last"])
def test_cuts_stay_ordered_even_when_every_cut_is_at_a_window_edge(pick):
    # Worst case for neighbouring windows: 10 s chunks with a window just under half.
    wav = FakeWav(100, sample_rate=1000)
    finder = (lambda s, r: 1) if pick == "first" else (lambda s, r: len(s) - 1)
    chunks = plan_chunks(wav, chunk_seconds=10, snap_window=4.99, find_gap=finder)
    _assert_contiguous(chunks, wav.num_samples)
    assert len(chunks) == 10


# --- tail, short and empty files -------------------------------------------


def test_short_tail_is_merged_into_the_previous_chunk():
    wav = FakeWav(130)
    calls = []
    chunks = plan_chunks(wav, 60, 5, find_gap=lambda *a: calls.append(a))
    assert _ranges(chunks) == [(0, 6000), (6000, 13000)]
    assert len(calls) == 1  # the dropped boundary is not even examined


def test_tail_of_a_quarter_chunk_is_kept():
    chunks = plan_chunks(FakeWav(135), 60, 5, find_gap=_never)
    assert _ranges(chunks) == [(0, 6000), (6000, 12000), (12000, 13500)]


def test_file_shorter_than_one_chunk_is_a_single_chunk():
    calls = []
    chunks = plan_chunks(FakeWav(20), 60, 5, find_gap=lambda *a: calls.append(a))
    assert _ranges(chunks) == [(0, 2000)]
    assert chunks[0].snapped is True
    assert calls == []


def test_file_of_exactly_one_chunk_is_a_single_chunk():
    assert _ranges(plan_chunks(FakeWav(60), 60, 5, find_gap=_never)) == [(0, 6000)]


def test_exact_multiple_of_the_chunk_length():
    chunks = plan_chunks(FakeWav(180), 60, 5, find_gap=_never)
    assert _ranges(chunks) == [(0, 6000), (6000, 12000), (12000, 18000)]


def test_empty_file_has_no_chunks():
    assert plan_chunks(FakeWav(0), find_gap=_never) == []


def test_times_derive_from_sample_indices_without_drift():
    rate = 16000
    wav = FakeWav(12 * 3600, sample_rate=rate)  # a 12 hour job
    chunks = plan_chunks(wav, chunk_seconds=60, snap_window=1, find_gap=_never)
    assert len(chunks) == 720
    last = chunks[-1]
    assert last.start_sample == 719 * 60 * rate
    assert last.start == 719 * 60.0
    assert last.end == 12 * 3600.0
    assert Chunk(0, 1, rate + 1, True, rate).start == 1 / rate


@pytest.mark.parametrize("seconds", [0, 4.9, 300.1, -60])
def test_chunk_seconds_is_validated(seconds):
    with pytest.raises(ValueError, match="chunk_seconds"):
        plan_chunks(FakeWav(100), chunk_seconds=seconds, find_gap=_never)


@pytest.mark.parametrize("window", [-1, 30, 40])
def test_snap_window_must_be_below_half_a_chunk(window):
    with pytest.raises(ValueError, match="snap_window"):
        plan_chunks(FakeWav(100), chunk_seconds=60, snap_window=window, find_gap=_never)


def test_chunk_dict_round_trip():
    chunk = Chunk(3, 160000, 320000, False, 16000)
    entry = chunk.to_dict()
    assert entry == {
        "index": 3,
        "start_sample": 160000,
        "end_sample": 320000,
        "start": 10.0,
        "end": 20.0,
        "snapped": False,
    }
    assert Chunk.from_dict(entry, 16000) == chunk


# --- default gap finder ----------------------------------------------------


def _fake_vad(monkeypatch, speech, seen=None):
    def get_speech_timestamps(samples, options, sampling_rate):
        if seen is not None:
            seen.append(options)
        return [{"start": s, "end": e} for s, e in speech]

    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps", get_speech_timestamps)


WINDOW = np.zeros(160000, dtype=np.float32)  # 10 s at 16 kHz


def test_default_finder_cuts_in_the_middle_of_the_longest_gap(monkeypatch):
    seen = []
    _fake_vad(monkeypatch, [(0, 40000), (48000, 100000), (120000, 160000)], seen)
    assert default_find_gap(WINDOW, 16000) == 110000  # gap 100000-120000 beats 40000-48000
    # The library default (2000 ms) would bridge every pause shorter than 2 s.
    assert seen[0].min_silence_duration_ms == 300


def test_default_finder_uses_edge_gaps_and_silent_windows(monkeypatch):
    _fake_vad(monkeypatch, [(60000, 160000)])
    assert default_find_gap(WINDOW, 16000) == 30000
    _fake_vad(monkeypatch, [])
    assert default_find_gap(WINDOW, 16000) == 80000


def test_default_finder_prefers_the_gap_nearest_the_boundary_on_ties(monkeypatch):
    _fake_vad(monkeypatch, [(0, 40000), (48000, 100000), (108000, 160000)])
    assert default_find_gap(WINDOW, 16000) == 104000  # 24000 from centre vs 36000


def test_default_finder_ignores_gaps_shorter_than_0_3_seconds(monkeypatch):
    _fake_vad(monkeypatch, [(0, 60000), (64000, 160000)])  # 0.25 s pause
    assert default_find_gap(WINDOW, 16000) is None


def test_default_finder_falls_back_to_the_quietest_frame_when_vad_fails(monkeypatch):
    def broken(*_args):
        raise RuntimeError("onnxruntime is unavailable")

    monkeypatch.setattr(chunking, "_vad_gap", broken)
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(32000) * 0.3).astype(np.float32)
    samples[16000:17600] = 0.0  # the 11th 100 ms frame
    assert default_find_gap(samples, 16000) == 16800


def test_quietest_frame_prefers_the_centre_when_everything_is_equally_quiet():
    cut = chunking._quietest_frame(np.zeros(32000, dtype=np.float32), 16000)
    assert abs(cut - 16000) <= 1600


def test_quietest_frame_needs_at_least_one_frame():
    assert chunking._quietest_frame(np.zeros(1000, dtype=np.float32), 16000) is None


def test_default_finder_only_supports_16khz_and_falls_back_otherwise():
    samples = np.ones(8000, dtype=np.float32) * 0.2
    samples[4000:4800] = 0.0
    # 8 kHz cannot go through Silero, so the RMS fallback answers (frame 5 of 800 samples).
    assert default_find_gap(samples, 8000) == 4400


def test_real_vad_runs_on_synthetic_audio(tmp_path):
    # Correctness of Silero's decisions on synthetic noise is not asserted; this only
    # proves the bundled model loads and the default finder yields a valid plan.
    rate = 16000
    rng = np.random.default_rng(1)
    t = np.arange(rate * 14) / rate
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    audio += 0.02 * rng.standard_normal(len(t))
    path = tmp_path / "synthetic.wav"
    write_pcm16_wav(path, audio.astype(np.float32), rate)

    with open_pcm16_mono(path) as wav:
        chunks = plan_chunks(wav, chunk_seconds=5, snap_window=1.5)

    _assert_contiguous(chunks, rate * 14)
    assert len(chunks) == 3
    for nominal, chunk in zip((5 * rate, 10 * rate), chunks):
        assert abs(chunk.end_sample - nominal) <= int(1.5 * rate)
