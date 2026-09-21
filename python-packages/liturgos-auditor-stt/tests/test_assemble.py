from datetime import datetime, timedelta, timezone

import pytest

from auditor_stt.serve.jobs.assemble import IncompleteJobError, assemble, progress
from auditor_stt.serve.jobs.chunking import Chunk

RATE = 16000
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _manifest(started_at=T0, planned=True, audio=True):
    # 150 s of audio in chunks of 60, 60 and 30 s.
    bounds = [(0, 60), (60, 120), (120, 150)]
    return {
        "params": {"language": "fi"},
        "started_at": started_at.isoformat() if started_at else None,
        "audio": {"wav_path": "audio.wav", "duration_seconds": 150.0, "sample_rate": RATE} if audio else None,
        "chunks": (
            [Chunk(i, s * RATE, e * RATE, True, RATE).to_dict() for i, (s, e) in enumerate(bounds)] if planned else None
        ),
    }


def _result(text, start, language="fi"):
    segments = [{"start": start, "end": start + 1.0, "text": text, "words": []}] if text else []
    return {"text": text, "language": language, "segments": segments}


RESULTS = {0: _result("Hyvää huomenta.", 0.0), 1: _result("Tervetuloa.", 60.0), 2: _result("Aamen.", 120.0)}


# --- assemble --------------------------------------------------------------


def test_assemble_concatenates_in_chunk_order():
    shuffled = {2: RESULTS[2], 0: RESULTS[0], 1: RESULTS[1]}
    result = assemble(_manifest(), shuffled)

    assert result["text"] == "Hyvää huomenta. Tervetuloa. Aamen."
    assert result["language"] == "fi"
    assert [s["start"] for s in result["segments"]] == [0.0, 60.0, 120.0]
    assert result["complete"] is True
    assert (result["chunks_done"], result["chunks_total"]) == (3, 3)
    assert result["duration_seconds"] == 150.0
    assert set(result) == {"text", "language", "segments", "complete", "chunks_done", "chunks_total", "duration_seconds"}


def test_assemble_joins_text_with_single_spaces_and_skips_silent_chunks():
    results = {
        0: {"text": "  eka  ", "language": "fi", "segments": []},
        1: {"text": "", "language": "fi", "segments": []},  # a silent minute
        2: {"text": "toka", "language": "fi", "segments": []},
    }
    assert assemble(_manifest(), results)["text"] == "eka toka"


def test_assemble_missing_chunk_raises_unless_partial():
    incomplete = {0: RESULTS[0], 2: RESULTS[2]}
    with pytest.raises(IncompleteJobError, match="2 of 3"):
        assemble(_manifest(), incomplete)

    partial = assemble(_manifest(), incomplete, partial=True)
    assert partial["complete"] is False
    assert (partial["chunks_done"], partial["chunks_total"]) == (2, 3)
    assert partial["text"] == "Hyvää huomenta. Aamen."
    assert [s["start"] for s in partial["segments"]] == [0.0, 120.0]
    assert partial["duration_seconds"] == 150.0  # of the whole file, not of what is done


def test_assemble_partial_of_a_finished_job_is_complete():
    assert assemble(_manifest(), RESULTS, partial=True)["complete"] is True


def test_assemble_language_is_the_first_non_null_chunk_language():
    results = {
        0: _result("a", 0.0, language=None),
        1: _result("b", 60.0, language="sv"),
        2: _result("c", 120.0, language="fi"),
    }
    assert assemble(_manifest(), results)["language"] == "sv"


def test_assemble_language_falls_back_to_the_requested_language():
    results = {i: _result("x", i * 60.0, language=None) for i in range(3)}
    assert assemble(_manifest(), results)["language"] == "fi"
    assert assemble(_manifest(), {}, partial=True)["language"] == "fi"


def test_assemble_ignores_unplanned_indices_and_unreadable_results():
    results = {**RESULTS, 3: _result("stray", 180.0), 1: None}
    with pytest.raises(IncompleteJobError):
        assemble(_manifest(), results)
    partial = assemble(_manifest(), results, partial=True)
    assert partial["chunks_done"] == 2
    assert "stray" not in partial["text"]


def test_assemble_before_the_plan_exists():
    manifest = _manifest(planned=False, audio=False)
    empty = assemble(manifest, {}, partial=True)
    assert empty["text"] == "" and empty["segments"] == []
    assert empty["complete"] is False
    assert (empty["chunks_done"], empty["chunks_total"]) == (0, 0)
    assert empty["duration_seconds"] == 0.0
    with pytest.raises(IncompleteJobError):
        assemble(manifest, {})


def test_assemble_duration_falls_back_to_the_last_chunk_end():
    assert assemble(_manifest(audio=False), RESULTS)["duration_seconds"] == 150.0


# --- progress --------------------------------------------------------------


def test_progress_before_anything_is_done():
    p = progress(_manifest(), 0, now=T0 + timedelta(seconds=5))
    assert p == {
        "progress": 0.0,
        "current_seconds": 0.0,
        "total_seconds": 150.0,
        "eta_seconds": None,
        "chunks_done": 0,
        "chunks_total": 3,
    }


def test_progress_and_eta_after_the_first_chunk():
    p = progress(_manifest(), 1, now=T0 + timedelta(seconds=30))
    assert p["progress"] == pytest.approx(40.0)
    assert p["current_seconds"] == 60.0
    assert p["total_seconds"] == 150.0
    # 60 s of audio took 30 s, and 90 s remain.
    assert p["eta_seconds"] == pytest.approx(45.0)
    assert (p["chunks_done"], p["chunks_total"]) == (1, 3)


def test_progress_when_finished():
    p = progress(_manifest(), 3, now=T0 + timedelta(seconds=100))
    assert p["progress"] == 100.0
    assert p["current_seconds"] == 150.0
    assert p["eta_seconds"] == 0.0


def test_eta_is_unknown_until_the_run_has_started():
    assert progress(_manifest(started_at=None), 1)["eta_seconds"] is None


def test_eta_excludes_chunks_finished_before_a_resume():
    manifest = _manifest()
    now = T0 + timedelta(seconds=30)
    # Chunk 0 was done before the restart; this run has only produced chunk 1.
    resumed = progress(manifest, 2, now=now, baseline_chunks=1)
    naive = progress(manifest, 2, now=now)
    assert resumed["eta_seconds"] == pytest.approx(30 / 60 * 30)
    assert naive["eta_seconds"] == pytest.approx(30 / 120 * 30)
    # Nothing new since the resume: not measurable yet.
    assert progress(manifest, 1, now=now, baseline_chunks=1)["eta_seconds"] is None


def test_progress_clamps_chunks_done():
    assert progress(_manifest(), 99)["chunks_done"] == 3
    assert progress(_manifest(), -4)["chunks_done"] == 0


def test_progress_without_a_plan():
    p = progress(_manifest(planned=False), 0)
    assert p["progress"] == 0.0
    assert p["chunks_total"] == 0
    assert p["total_seconds"] == 150.0
    assert p["eta_seconds"] is None


def test_progress_treats_a_naive_now_as_utc_and_ignores_a_clock_that_ran_backwards():
    assert progress(_manifest(), 1, now=datetime(2026, 1, 1, 12, 0, 30))["eta_seconds"] == pytest.approx(45.0)
    assert progress(_manifest(), 1, now=T0 - timedelta(seconds=5))["eta_seconds"] is None


def test_progress_now_defaults_to_the_current_time():
    started = datetime.now(timezone.utc) - timedelta(seconds=10)
    p = progress(_manifest(started_at=started), 1)
    assert p["eta_seconds"] is not None and p["eta_seconds"] > 0
