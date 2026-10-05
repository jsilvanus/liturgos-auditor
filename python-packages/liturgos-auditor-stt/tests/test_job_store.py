import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from auditor_stt.serve.jobs import store as store_module
from auditor_stt.serve.jobs.store import InvalidJobIdError, JobNotFoundError, JobStore

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, **delta):
        self.now += timedelta(**delta)


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(store_module, "_utcnow", clock)
    return clock


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs")


def _create(store, **params):
    return store.create(params=params, source={"kind": "path", "path": "/media/sermon.mp4"})


def _tmp_files(store):
    return sorted(p.name for p in store.root.rglob("*.tmp"))


# --- create / load / update ------------------------------------------------


def test_create_and_load(store):
    manifest = _create(store, language="fi", client_ref="project-7")

    assert re.fullmatch(r"[0-9a-f]{32}", manifest["id"])
    assert manifest["status"] == "queued"
    assert manifest["params"] == {
        "language": "fi",
        "chunk_seconds": 60.0,
        "word_timestamps": True,
        "prompt": None,
        "client_ref": "project-7",
    }
    assert manifest["source"] == {"kind": "path", "path": "/media/sermon.mp4"}
    assert manifest["created_at"] == manifest["updated_at"]
    for field in ("started_at", "finished_at", "audio", "chunks", "error"):
        assert manifest[field] is None
    assert manifest["current_seconds"] == 0.0
    assert manifest["cancel_requested"] is False

    assert store.load(manifest["id"]) == manifest
    assert (store.job_dir(manifest["id"]) / "manifest.json").is_file()
    assert (store.job_dir(manifest["id"]) / "chunks").is_dir()


def test_create_with_explicit_id_and_source_validation(store):
    job_id = "0123456789abcdef0123456789abcdef"
    assert store.create(params={}, source={"kind": "upload", "path": None}, job_id=job_id)["id"] == job_id
    with pytest.raises(FileExistsError):
        store.create(params={}, source={"kind": "upload", "path": None}, job_id=job_id)
    with pytest.raises(ValueError, match="source kind"):
        store.create(params={}, source={"kind": "ftp", "path": "http://x"})


def test_load_unknown_job(store):
    with pytest.raises(JobNotFoundError):
        store.load("0" * 32)


def test_update_merges_fields_and_bumps_updated_at(store, clock):
    job_id = _create(store)["id"]
    clock.advance(seconds=5)
    audio = {"wav_path": "audio.wav", "duration_seconds": 12.5, "sample_rate": 16000}

    updated = store.update(job_id, audio=audio, phase="planning", current_seconds=3.5)

    assert updated["audio"] == audio
    assert updated["phase"] == "planning"
    assert updated["params"]["language"] == "fi"  # untouched fields survive
    assert updated["updated_at"] > updated["created_at"]
    assert store.load(job_id) == updated


def test_update_rejects_protected_and_unknown_fields(store):
    job_id = _create(store)["id"]
    for field in ("id", "status", "created_at", "updated_at", "started_at", "finished_at", "typo"):
        with pytest.raises(ValueError, match=field):
            store.update(job_id, **{field: "x"})
    assert store.load(job_id)["status"] == "queued"


def test_update_unknown_job(store):
    with pytest.raises(JobNotFoundError):
        store.update("0" * 32, phase="x")


def test_finnish_text_is_stored_readably(store):
    job_id = _create(store, prompt="Hyvää huomenta, seurakunta")["id"]
    raw = (store.job_dir(job_id) / "manifest.json").read_text(encoding="utf-8")
    assert "Hyvää huomenta" in raw


# --- status ----------------------------------------------------------------


def test_set_status_maintains_started_and_finished_at(store, clock):
    job_id = _create(store)["id"]

    clock.advance(seconds=10)
    running = store.set_status(job_id, "running")
    assert running["status"] == "running"
    assert running["started_at"] == "2026-01-01T12:00:10.000000+00:00"
    assert running["finished_at"] is None

    clock.advance(seconds=50)
    done = store.set_status(job_id, "completed")
    assert done["started_at"] == running["started_at"]
    assert done["finished_at"] == "2026-01-01T12:01:00.000000+00:00"


def test_set_status_failed_carries_the_error_and_running_clears_it(store):
    job_id = _create(store)["id"]
    failed = store.set_status(job_id, "failed", error="ffmpeg exited with status 1")
    assert failed["error"] == "ffmpeg exited with status 1"
    assert failed["finished_at"] is not None

    rerun = store.set_status(job_id, "running")
    assert rerun["error"] is None
    assert rerun["finished_at"] is None


def test_requeue_clears_run_timestamps(store):
    job_id = _create(store)["id"]
    store.set_status(job_id, "running")
    requeued = store.set_status(job_id, "queued")
    assert requeued["started_at"] is None and requeued["finished_at"] is None


def test_set_status_rejects_unknown_status(store):
    job_id = _create(store)["id"]
    with pytest.raises(ValueError, match="status"):
        store.set_status(job_id, "paused")


# --- atomic writes ---------------------------------------------------------


def test_no_tmp_files_are_left_behind(store):
    job_id = _create(store)["id"]
    store.update(job_id, phase="a")
    store.set_status(job_id, "running")
    store.write_chunk(job_id, 0, {"text": "x", "segments": []})
    assert _tmp_files(store) == []


def test_failed_write_keeps_the_old_manifest_and_cleans_up(store, monkeypatch):
    job_id = _create(store)["id"]
    store.update(job_id, phase="before")

    def broken_replace(source, target):
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", broken_replace)
        with pytest.raises(OSError, match="disk full"):
            store.update(job_id, phase="after")
        with pytest.raises(OSError):
            store.write_chunk(job_id, 0, {"text": "lost"})

    assert store.load(job_id)["phase"] == "before"
    assert store.done_indices(job_id) == []
    assert _tmp_files(store) == []


def test_failed_create_leaves_no_job_behind(store, monkeypatch):
    def broken_replace(source, target):
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", broken_replace)
        with pytest.raises(OSError):
            _create(store)

    assert list(store.root.iterdir()) == []


def test_replace_is_retried_when_windows_holds_the_file(store, monkeypatch):
    job_id = _create(store)["id"]
    real_replace = os.replace
    attempts = []

    def flaky_replace(source, target):
        attempts.append(source)
        if len(attempts) < 3:
            raise PermissionError("sharing violation")
        real_replace(source, target)

    monkeypatch.setattr(store_module.time, "sleep", lambda _s: None)
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", flaky_replace)
        store.update(job_id, phase="retried")

    assert len(attempts) == 3
    assert store.load(job_id)["phase"] == "retried"
    assert _tmp_files(store) == []


def test_corrupt_manifest_raises_on_load(store):
    job_id = _create(store)["id"]
    (store.job_dir(job_id) / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load(job_id)


# --- job ids ---------------------------------------------------------------

BAD_IDS = ["../x", "..", "", ".", "a/b", "a\\b", "A" * 32, "g" * 32, "a" * 31, "a" * 33, "0" * 32 + "/../..", 42]

PUBLIC_CALLS = {
    "job_dir": lambda s, i: s.job_dir(i),
    "source_dir": lambda s, i: s.source_dir(i),
    "audio_path": lambda s, i: s.audio_path(i),
    "load": lambda s, i: s.load(i),
    "update": lambda s, i: s.update(i, phase="x"),
    "set_status": lambda s, i: s.set_status(i, "running"),
    "request_cancel": lambda s, i: s.request_cancel(i),
    "cancel_requested": lambda s, i: s.cancel_requested(i),
    "write_chunk": lambda s, i: s.write_chunk(i, 0, {}),
    "read_chunk": lambda s, i: s.read_chunk(i, 0),
    "done_indices": lambda s, i: s.done_indices(i),
    "delete": lambda s, i: s.delete(i),
    "remove_working_files": lambda s, i: s.remove_working_files(i),
    "create": lambda s, i: s.create(params={}, source={"kind": "path", "path": "x"}, job_id=i),
}


@pytest.mark.parametrize("call", sorted(PUBLIC_CALLS))
def test_every_public_call_rejects_malformed_ids(store, tmp_path, call):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "keep").write_text("precious")

    for bad_id in BAD_IDS:
        with pytest.raises(InvalidJobIdError):
            PUBLIC_CALLS[call](store, bad_id)

    assert (tmp_path / "x" / "keep").read_text() == "precious"
    assert list(store.root.iterdir()) == []


def test_none_is_not_a_job_id_for_lookups(store):
    with pytest.raises(InvalidJobIdError):
        store.load(None)


def test_invalid_id_error_is_a_not_found_and_a_value_error():
    assert issubclass(InvalidJobIdError, JobNotFoundError)
    assert issubclass(InvalidJobIdError, ValueError)


# --- listing ---------------------------------------------------------------


def test_list_is_oldest_first_and_skips_damaged_jobs(store, clock):
    ids = []
    for _ in range(3):
        ids.append(_create(store)["id"])
        clock.advance(seconds=1)

    (store.root / "not-a-job").mkdir()
    (store.root / ("f" * 32)).mkdir()  # job-shaped directory without a manifest
    broken = _create(store)["id"]
    (store.job_dir(broken) / "manifest.json").write_text("{", encoding="utf-8")
    (store.root / "stray.txt").write_text("x")

    assert [m["id"] for m in store.list()] == ids


def test_list_of_an_empty_store(store):
    assert store.list() == []


# --- cancellation ----------------------------------------------------------


def test_cancel_flag(store):
    job_id = _create(store)["id"]
    assert store.cancel_requested(job_id) is False
    assert store.request_cancel(job_id)["cancel_requested"] is True
    assert store.cancel_requested(job_id) is True
    assert store.load(job_id)["cancel_requested"] is True


def test_deleted_job_counts_as_cancelled_for_the_runner(store):
    job_id = _create(store)["id"]
    store.delete(job_id)
    assert store.cancel_requested(job_id) is True


# --- chunk results ---------------------------------------------------------


def test_chunk_round_trip_and_done_indices(store):
    job_id = _create(store)["id"]
    result = {"text": "Hyvää huomenta", "segments": [{"start": 60.0, "end": 61.0, "text": "Hyvää huomenta"}]}

    for index in (10, 2, 0):
        store.write_chunk(job_id, index, result)

    assert store.read_chunk(job_id, 2) == result
    assert store.done_indices(job_id) == [0, 2, 10]
    assert (store.job_dir(job_id) / "chunks" / "00010.json").is_file()


def test_chunk_can_be_rewritten(store):
    job_id = _create(store)["id"]
    store.write_chunk(job_id, 0, {"text": "old"})
    store.write_chunk(job_id, 0, {"text": "new"})
    assert store.read_chunk(job_id, 0) == {"text": "new"}


def test_corrupt_and_stray_chunk_files_are_not_done(store):
    job_id = _create(store)["id"]
    store.write_chunk(job_id, 0, {"text": "ok"})
    chunks = store.job_dir(job_id) / "chunks"
    (chunks / "00001.json").write_text('{"text": "cut off', encoding="utf-8")  # crash mid-write
    (chunks / "00002.json").write_bytes(b"")
    (chunks / "00003.json").write_text("[1, 2]", encoding="utf-8")  # valid JSON, wrong shape
    (chunks / "00004.json.tmp").write_text("{}", encoding="utf-8")
    (chunks / "notes.txt").write_text("hi", encoding="utf-8")
    (chunks / "00005.json").write_bytes(b"\xff\xfe\x00")

    assert store.done_indices(job_id) == [0]
    assert store.read_chunk(job_id, 1) is None
    assert store.read_chunk(job_id, 3) is None
    assert store.read_chunk(job_id, 99) is None


def test_chunk_calls_on_unknown_jobs_and_bad_indices(store):
    unknown = "0" * 32
    with pytest.raises(JobNotFoundError):
        store.write_chunk(unknown, 0, {})
    with pytest.raises(JobNotFoundError):
        store.done_indices(unknown)
    job_id = _create(store)["id"]
    for bad in (-1, "3", 1.5, None):
        with pytest.raises(ValueError):
            store.write_chunk(job_id, bad, {})


# --- working files ---------------------------------------------------------


def test_working_file_helpers_do_not_create_anything(store):
    job_id = _create(store)["id"]
    assert store.source_dir(job_id) == store.job_dir(job_id) / "source"
    assert store.audio_path(job_id) == store.job_dir(job_id) / "audio.wav"
    assert not store.source_dir(job_id).exists()
    assert not store.audio_path(job_id).exists()


def test_remove_working_files_keeps_results_and_foreign_sources(store, tmp_path):
    media = tmp_path / "media.mp4"
    media.write_bytes(b"shared volume file")
    job = store.create(params={}, source={"kind": "path", "path": media})
    job_id = job["id"]
    store.source_dir(job_id).mkdir()
    (store.source_dir(job_id) / "upload.mp4").write_bytes(b"upload")
    store.audio_path(job_id).write_bytes(b"wav")
    store.write_chunk(job_id, 0, {"text": "kept"})

    store.remove_working_files(job_id)
    store.remove_working_files(job_id)  # nothing left to remove is fine

    assert not store.source_dir(job_id).exists()
    assert not store.audio_path(job_id).exists()
    assert media.read_bytes() == b"shared volume file"
    assert store.read_chunk(job_id, 0) == {"text": "kept"}
    assert store.load(job_id)["id"] == job_id


def test_delete_removes_everything_and_reports_whether_it_existed(store):
    job_id = _create(store)["id"]
    store.write_chunk(job_id, 0, {"text": "x"})
    assert store.delete(job_id) is True
    assert not store.job_dir(job_id).exists()
    assert store.delete(job_id) is False
    assert store.list() == []


# --- recovery --------------------------------------------------------------


def test_recover_requeues_running_jobs_and_lists_queued_oldest_first(store, clock):
    a = _create(store)["id"]
    clock.advance(seconds=1)
    b = _create(store)["id"]
    clock.advance(seconds=1)
    c = _create(store)["id"]
    clock.advance(seconds=1)
    d = _create(store)["id"]

    store.set_status(a, "running")
    store.update(a, phase="transcribing", current_seconds=120.0)
    store.write_chunk(a, 0, {"text": "done before the crash"})
    store.set_status(c, "completed")
    store.set_status(d, "running")

    assert store.recover() == [a, b, d]

    recovered = store.load(a)
    assert recovered["status"] == "queued"
    assert recovered["started_at"] is None
    assert recovered["phase"] == "queued"
    assert store.load(d)["status"] == "queued"
    assert store.load(c)["status"] == "completed"
    assert store.done_indices(a) == [0]  # results survive for the resume


def test_recover_on_an_empty_or_damaged_store(store):
    assert store.recover() == []
    broken = _create(store)["id"]
    (store.job_dir(broken) / "manifest.json").write_text("nope", encoding="utf-8")
    assert store.recover() == []


# --- retention -------------------------------------------------------------


def test_purge_expired_only_removes_old_finished_jobs(store, clock):
    old_done, old_failed, old_cancelled = (_create(store)["id"] for _ in range(3))
    old_running, old_queued = (_create(store)["id"] for _ in range(2))
    store.set_status(old_done, "completed")
    store.set_status(old_failed, "failed", error="boom")
    store.set_status(old_cancelled, "cancelled")
    store.set_status(old_running, "running")

    clock.advance(hours=47)
    recent = _create(store)["id"]
    store.set_status(recent, "completed")

    purged = store.purge_expired(ttl_hours=24, now=T0 + timedelta(hours=48))

    assert sorted(purged) == sorted([old_done, old_failed, old_cancelled])
    remaining = {m["id"] for m in store.list()}
    assert remaining == {old_running, old_queued, recent}
    for gone in purged:
        assert not store.job_dir(gone).exists()


def test_purge_expired_measures_from_the_end_of_the_job(store, clock):
    job_id = _create(store)["id"]
    store.set_status(job_id, "running")
    clock.advance(hours=30)  # a long job: created 30 h before it finished
    store.set_status(job_id, "completed")

    assert store.purge_expired(24, now=T0 + timedelta(hours=40)) == []
    assert store.purge_expired(24, now=T0 + timedelta(hours=60)) == [job_id]


def test_purge_expired_defaults_to_now_and_rejects_negative_ttl(store):
    job_id = _create(store)["id"]
    store.set_status(job_id, "completed")
    assert store.purge_expired(24) == []
    assert store.purge_expired(0, now=datetime.now(timezone.utc) + timedelta(seconds=1)) == [job_id]
    with pytest.raises(ValueError):
        store.purge_expired(-1)


# --- concurrency -----------------------------------------------------------


def test_concurrent_updates_do_not_lose_fields(store):
    job_id = _create(store)["id"]
    rounds = 40
    barrier = threading.Barrier(5)

    def loop(make_fields):
        barrier.wait()
        for i in range(rounds):
            store.update(job_id, **make_fields(i))

    def statuses():
        barrier.wait()
        for i in range(rounds):
            store.set_status(job_id, "running" if i % 2 == 0 else "queued")
        store.set_status(job_id, "running")

    def cancel():
        barrier.wait()
        for _ in range(rounds // 4):
            store.load(job_id)
        store.request_cancel(job_id)

    workers = [
        lambda: loop(lambda i: {"phase": f"phase-{i}"}),
        lambda: loop(lambda i: {"current_seconds": float(i)}),
        lambda: loop(lambda i: {"audio": {"wav_path": "a.wav", "duration_seconds": float(i), "sample_rate": 16000}}),
        statuses,
        cancel,
    ]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for future in [pool.submit(worker) for worker in workers]:
            future.result()

    final = store.load(job_id)
    assert final["phase"] == f"phase-{rounds - 1}"
    assert final["current_seconds"] == float(rounds - 1)
    assert final["audio"]["duration_seconds"] == float(rounds - 1)
    assert final["status"] == "running"
    assert final["cancel_requested"] is True
    assert final["params"]["language"] == "fi"
    assert _tmp_files(store) == []
    json.loads((store.job_dir(job_id) / "manifest.json").read_text(encoding="utf-8"))
