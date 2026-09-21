import asyncio
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import pytest

from auditor_stt.serve.audio import MediaDecodeError, NormalizationCancelledError
from auditor_stt.serve.jobs.assemble import assemble
from auditor_stt.serve.jobs.runner import JobRunner, RunnerConfig
from auditor_stt.serve.jobs.store import JobStore
from auditor_stt.serve.jobs.wav import write_pcm16_wav
from auditor_stt.serve.queue import InferenceQueue

RATE = 16000
PARAMS = {"chunk_seconds": 5.0, "language": "fi"}
# A 30 s file in 5 s chunks: six chunks starting at 0, 5, ..., 25. "Chunk 3" starts at 15 s.
OFFSETS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def _result(offset, duration):
    text = f"chunk at {offset:g}"
    word = {"start": offset, "end": offset + duration, "text": text, "probability": 0.9}
    segment = {"start": offset, "end": offset + duration, "text": text, "avg_logprob": -0.1, "no_speech_prob": 0.01}
    return {"text": text, "language": "fi", "segments": [{**segment, "words": [word]}]}


class _Host:
    """Stub model. `hook(offset)` runs in the model thread before the result is returned."""

    def __init__(self):
        self.loaded = True
        self.calls = []
        self.hook = None

    @property
    def offsets(self):
        return [call["offset"] for call in self.calls]

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
                "offset": time_offset,
                "prompt": prompt,
                "vad": vad,
                "language": language,
                "condition": condition_on_previous_text,
                "word_timestamps": word_timestamps,
            }
        )
        if self.hook is not None:
            self.hook(time_offset)
        return _result(time_offset, len(samples) / RATE)


def _no_gap(_samples, _rate):
    return None


def _copy_normalize(source, wav_path, *, cancel=None):
    shutil.copyfile(source, wav_path)


def _write_wav(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm16_wav(path, np.zeros(int(seconds * RATE), dtype=np.float32), RATE)
    return path


def _runner(store, host, queue=None, *, normalize=_copy_normalize, **config):
    config = RunnerConfig(retry_backoff=(0.0, 0.0), model_poll_seconds=0.01, **config)
    return JobRunner(store, lambda: host, queue or InferenceQueue(), config, normalize=normalize, find_gap=_no_gap)


def _submit_path(store, runner, source, **params):
    manifest = store.create(params={**PARAMS, **params}, source={"kind": "path", "path": source})
    runner.submit(manifest["id"])
    return manifest["id"]


def _submit_upload(store, runner, seconds=30, name="talk.wav"):
    manifest = store.create(params=PARAMS, source={"kind": "upload", "path": None})
    job_id = manifest["id"]
    target = _write_wav(store.source_dir(job_id) / name, seconds)
    store.update(job_id, source={"kind": "upload", "path": str(target)})
    runner.submit(job_id)
    return job_id


async def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.005)


def _status(store, job_id):
    try:
        return store.load(job_id)["status"]
    except LookupError:
        return None


async def _wait_status(store, job_id, *statuses):
    await _wait_for(lambda: _status(store, job_id) in statuses)


async def _settled(runner):
    """Until the worker is done with everything, including the cleanup after a job."""
    await _wait_for(lambda: runner._run is None and runner._pending.empty())


async def _finish(store, runner, job_id, *statuses):
    await _wait_status(store, job_id, *(statuses or ("completed", "failed", "cancelled")))
    await _settled(runner)
    return store.load(job_id)


@asynccontextmanager
async def _running(runner):
    await runner.start()
    try:
        yield runner
    finally:
        await runner.stop()


def _result_of(store, job_id, partial=False):
    manifest = store.load(job_id)
    total = len(manifest["chunks"])
    return assemble(manifest, {i: store.read_chunk(job_id, i) for i in range(total)}, partial=partial)


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs")


@pytest.fixture
def media(tmp_path):
    return _write_wav(tmp_path / "media" / "talk.wav", 30)


# --- a normal run ----------------------------------------------------------------


async def test_a_job_runs_to_completion_and_keeps_only_its_results(store, media):
    host = _Host()
    runner = _runner(store, host)
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "completed"
    assert manifest["phase"] == "completed"
    assert manifest["error"] is None
    assert manifest["started_at"] and manifest["finished_at"]
    assert manifest["current_seconds"] == 30.0
    assert manifest["audio"]["duration_seconds"] == pytest.approx(30.0)
    assert [chunk["start"] for chunk in manifest["chunks"]] == OFFSETS  # the persisted plan
    assert store.done_indices(job_id) == [0, 1, 2, 3, 4, 5]

    assert host.offsets == OFFSETS
    assert all(call["condition"] is False and call["language"] == "fi" for call in host.calls)
    assert all(call["vad"] is True and call["word_timestamps"] is True for call in host.calls)
    assert host.calls[0]["prompt"] is None
    assert host.calls[1]["prompt"] == "chunk at 0"  # the previous chunk's text carries over

    result = _result_of(store, job_id)
    assert result["complete"] is True
    assert [segment["start"] for segment in result["segments"]] == OFFSETS  # absolute time
    assert result["text"] == " ".join(f"chunk at {offset:g}" for offset in OFFSETS)

    assert not store.audio_path(job_id).exists()  # the normalised WAV is gone
    assert media.exists()  # a caller's file on the media volume is never deleted


async def test_an_uploaded_copy_is_deleted_when_the_job_ends(store):
    runner = _runner(store, _Host())
    job_id = _submit_upload(store, runner)
    source_dir = store.source_dir(job_id)
    assert source_dir.is_dir()

    async with _running(runner):
        await _finish(store, runner, job_id, "completed")

    assert not source_dir.exists()
    assert not store.audio_path(job_id).exists()
    assert store.done_indices(job_id) == [0, 1, 2, 3, 4, 5]  # results and manifest stay


async def test_jobs_run_one_at_a_time_in_submission_order(store, tmp_path):
    host = _Host()
    runner = _runner(store, host)
    first = _submit_path(store, runner, _write_wav(tmp_path / "media" / "a.wav", 30))
    second = _submit_path(store, runner, _write_wav(tmp_path / "media" / "b.wav", 20))

    async with _running(runner):
        await _finish(store, runner, second, "completed")

    assert host.offsets == OFFSETS + OFFSETS[:4]  # never interleaved
    assert store.load(first)["finished_at"] <= store.load(second)["started_at"]


async def test_batch_vad_and_carry_over_follow_the_config(store, media):
    host = _Host()
    runner = _runner(store, host, batch_vad=False, carry_chars=0)
    job_id = _submit_path(store, runner, media, prompt="kirkko")

    async with _running(runner):
        await _finish(store, runner, job_id, "completed")

    assert all(call["vad"] is False for call in host.calls)
    assert all(call["prompt"] == "kirkko" for call in host.calls)  # no carried text


def test_config_defaults_and_environment(monkeypatch):
    for name in ("AUDITOR_STT_JOB_TTL_HOURS", "AUDITOR_STT_CARRY_CONTEXT_CHARS", "AUDITOR_STT_BATCH_VAD"):
        monkeypatch.delenv(name, raising=False)
    config = RunnerConfig.from_env()
    assert (config.ttl_hours, config.carry_chars, config.batch_vad) == (72.0, 200, True)

    monkeypatch.setenv("AUDITOR_STT_JOB_TTL_HOURS", "1.5")
    monkeypatch.setenv("AUDITOR_STT_CARRY_CONTEXT_CHARS", "0")
    monkeypatch.setenv("AUDITOR_STT_BATCH_VAD", "off")
    config = RunnerConfig.from_env()
    assert (config.ttl_hours, config.carry_chars, config.batch_vad) == (1.5, 0, False)

    monkeypatch.setenv("AUDITOR_STT_BATCH_VAD", "1")
    assert RunnerConfig.from_env().batch_vad is True


# --- abort and resume: the reason for chunking ----------------------------------


async def test_a_stopped_job_resumes_with_only_the_unfinished_chunks(tmp_path, media):
    # An uninterrupted run is the reference for the resumed one.
    reference_store = JobStore(tmp_path / "reference")
    reference_runner = _runner(reference_store, _Host())
    reference_id = _submit_path(reference_store, reference_runner, media)
    async with _running(reference_runner):
        await _finish(reference_store, reference_runner, reference_id, "completed")
    reference = _result_of(reference_store, reference_id)

    store = JobStore(tmp_path / "jobs")
    release = threading.Event()
    first_host = _Host()
    first_host.hook = lambda offset: offset == 15.0 and release.wait(10)  # chunk 3 hangs (a crash, in effect)
    normalize_calls = []

    def counting_normalize(source, wav_path, *, cancel=None):
        normalize_calls.append(source)
        _copy_normalize(source, wav_path)

    first = _runner(store, first_host, normalize=counting_normalize)
    job_id = _submit_path(store, first, media)
    try:
        await first.start()
        await _wait_for(lambda: 15.0 in first_host.offsets)
        assert store.done_indices(job_id) == [0, 1, 2]

        started = time.monotonic()
        await first.stop()
        assert time.monotonic() - started < 2.0  # does not wait for the hanging chunk
    finally:
        release.set()  # let the abandoned model thread end; its result is discarded
    await asyncio.sleep(0.05)

    interrupted = store.load(job_id)
    assert interrupted["status"] == "running"  # interrupted, not failed: the next start resumes it
    assert store.done_indices(job_id) == [0, 1, 2]  # chunk 3 was never persisted
    plan = interrupted["chunks"]
    kept = {index: store.read_chunk(job_id, index) for index in range(3)}

    # A new runner over the same store, as after a process restart.
    second_host = _Host()
    baselines = []
    second_host.hook = lambda offset: baselines.append(second.baseline_chunks(job_id))
    second = _runner(store, second_host, normalize=counting_normalize)
    async with _running(second):
        manifest = await _finish(store, second, job_id, "completed")

    assert first_host.offsets == [0.0, 5.0, 10.0, 15.0]
    assert second_host.offsets == [15.0, 20.0, 25.0]  # chunks 0-2 were not transcribed again
    assert second_host.calls[0]["prompt"] == "chunk at 10"  # carry-over read back from the saved chunk 2
    assert baselines == [3, 3, 3]  # ETA counts only the work of this run
    assert len(normalize_calls) == 1  # the normalised audio was reused

    assert manifest["chunks"] == plan  # the plan is never recomputed
    assert {index: store.read_chunk(job_id, index) for index in range(3)} == kept
    assert _result_of(store, job_id) == reference


async def test_a_job_left_running_by_a_crash_is_picked_up_at_start(store, media):
    crashed = store.create(params=PARAMS, source={"kind": "path", "path": media})["id"]
    store.set_status(crashed, "running")
    waiting = store.create(params=PARAMS, source={"kind": "path", "path": media})["id"]  # queued, never submitted
    host = _Host()

    async with _running(_runner(store, host)) as runner:
        await _wait_status(store, waiting, "completed")
        await _settled(runner)

    assert store.load(crashed)["status"] == "completed"
    assert store.load(crashed)["started_at"] <= store.load(waiting)["started_at"]  # oldest first
    assert host.offsets == OFFSETS + OFFSETS


async def test_start_removes_working_files_a_crash_left_behind(store):
    job_id = store.create(params=PARAMS, source={"kind": "upload", "path": None})["id"]
    store.set_status(job_id, "completed")
    _write_wav(store.source_dir(job_id) / "talk.wav", 1)
    _write_wav(store.audio_path(job_id), 1)

    async with _running(_runner(store, _Host())):
        pass

    assert not store.source_dir(job_id).exists()
    assert not store.audio_path(job_id).exists()
    assert store.load(job_id)["status"] == "completed"


# --- cancel and delete -----------------------------------------------------------


async def test_cancel_is_honoured_between_chunks(store, media):
    host = _Host()
    runner = _runner(store, host)
    job_id = _submit_path(store, runner, media)
    host.hook = lambda offset: offset == 10.0 and store.request_cancel(job_id)  # arrives during chunk 2

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "cancelled"
    assert manifest["phase"] == "cancelled"
    assert manifest["error"] is None
    assert host.offsets == [0.0, 5.0, 10.0]  # the chunk in flight completes; nothing after it starts
    assert store.done_indices(job_id) == [0, 1, 2]
    partial = _result_of(store, job_id, partial=True)
    assert partial["complete"] is False and partial["chunks_done"] == 3
    assert not store.audio_path(job_id).exists()
    assert media.exists()


async def test_a_job_cancelled_before_it_starts_never_touches_the_model(store, media):
    host = _Host()
    runner = _runner(store, host)
    job_id = _submit_path(store, runner, media)
    store.request_cancel(job_id)

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "cancelled"
    assert host.calls == []


async def test_discard_deletes_a_running_job_once_it_has_stopped(store, media):
    release = threading.Event()
    host = _Host()
    host.hook = lambda offset: offset == 5.0 and release.wait(10)
    runner = _runner(store, host)
    job_id = _submit_path(store, runner, media)

    try:
        async with _running(runner):
            await _wait_for(lambda: 5.0 in host.offsets)
            assert runner.discard("0" * 32) is False  # not the job in flight
            store.request_cancel(job_id)  # what DELETE does first
            assert runner.discard(job_id) is True
            assert store.job_dir(job_id).is_dir()  # still there while the chunk runs
            release.set()

            await _wait_for(lambda: not store.job_dir(job_id).exists())
            await _settled(runner)
    finally:
        release.set()

    assert host.offsets == [0.0, 5.0]
    assert store.list() == []
    assert media.exists()
    assert runner.discard(job_id) is False  # nothing left to discard


async def test_discard_interrupts_normalisation(store, media):
    started = threading.Event()

    def slow_normalize(source, wav_path, *, cancel=None):
        started.set()
        assert cancel.wait(10), "cancel was never signalled"
        raise NormalizationCancelledError()

    runner = _runner(store, _Host(), normalize=slow_normalize)
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        await _wait_for(started.is_set)
        assert runner.discard(job_id) is True
        await _wait_for(lambda: not store.job_dir(job_id).exists())
        await _settled(runner)

    assert media.exists()


async def test_stop_interrupts_normalisation_and_the_job_resumes_later(store, media):
    started = threading.Event()
    seen_cancel = threading.Event()

    def slow_normalize(source, wav_path, *, cancel=None):
        started.set()
        if cancel.wait(10):
            seen_cancel.set()
        raise NormalizationCancelledError()

    first = _runner(store, _Host(), normalize=slow_normalize)
    job_id = _submit_path(store, first, media)
    await first.start()
    await _wait_for(started.is_set)
    await first.stop()
    await _wait_for(seen_cancel.is_set)  # the ffmpeg run was told to stop, not left running
    assert store.load(job_id)["status"] == "running"

    host = _Host()
    async with _running(_runner(store, host)) as second:
        await _finish(store, second, job_id, "completed")
    assert host.offsets == OFFSETS


async def test_a_job_directory_removed_under_the_runner_does_not_crash_it(store, media, tmp_path):
    host = _Host()
    runner = _runner(store, host)
    doomed = _submit_path(store, runner, media)

    def vanish(offset):
        if offset == 10.0:
            os.unlink(store.job_dir(doomed) / "manifest.json")
            shutil.rmtree(store.job_dir(doomed) / "chunks")

    host.hook = vanish

    async with _running(runner):
        await _wait_for(lambda: not store.job_dir(doomed).exists())
        await _settled(runner)
        host.hook = None
        survivor = _submit_path(store, runner, media)
        await _finish(store, runner, survivor, "completed")  # the worker is still alive

    assert not store.job_dir(doomed).exists()
    assert host.offsets == [0.0, 5.0, 10.0] + OFFSETS  # nothing was attempted after the directory vanished


# --- failures --------------------------------------------------------------------


async def test_a_chunk_is_retried_twice_and_then_the_job_fails_keeping_its_partial_results(store):
    host = _Host()
    failures = []

    def flaky(offset):
        if offset == 15.0 and len(failures) < 3:
            failures.append(offset)
            raise RuntimeError("model exploded")

    host.hook = flaky
    runner = _runner(store, host)
    job_id = _submit_upload(store, runner)

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

        assert manifest["status"] == "failed"
        assert manifest["phase"] == "failed"
        assert manifest["error"] == "Chunk 3 failed after 3 attempts: RuntimeError: model exploded"
        assert host.offsets == [0.0, 5.0, 10.0, 15.0, 15.0, 15.0]  # the attempt plus two retries
        assert store.done_indices(job_id) == [0, 1, 2]  # completed chunks are kept
        partial = _result_of(store, job_id, partial=True)
        assert partial["complete"] is False and partial["text"] == "chunk at 0 chunk at 5 chunk at 10"
        assert not store.source_dir(job_id).exists() and not store.audio_path(job_id).exists()

        # A failed job does not take the worker down with it (the model has recovered by now).
        survivor = _submit_upload(store, runner)
        await _finish(store, runner, survivor, "completed")


async def test_a_transient_chunk_failure_is_retried_transparently(store, media):
    host = _Host()
    failures = []

    def flaky(offset):
        if offset == 10.0 and not failures:
            failures.append(offset)
            raise RuntimeError("out of memory")

    host.hook = flaky
    runner = _runner(store, host)
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "completed" and manifest["error"] is None
    assert host.offsets == [0.0, 5.0, 10.0, 10.0, 15.0, 20.0, 25.0]
    assert _result_of(store, job_id)["complete"] is True


async def test_a_media_decode_failure_fails_the_job_without_touching_the_source(store, media):
    def broken_normalize(source, wav_path, *, cancel=None):
        raise MediaDecodeError("Could not decode audio from talk.wav: Invalid data found")

    host = _Host()
    runner = _runner(store, host, normalize=broken_normalize)
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "failed"
    assert manifest["error"] == "Could not decode audio from talk.wav: Invalid data found"
    assert host.calls == []
    assert media.exists()


async def test_media_without_audio_fails_the_job(store, tmp_path):
    empty = tmp_path / "media" / "empty.wav"
    empty.parent.mkdir()
    write_pcm16_wav(empty, np.zeros(0, dtype=np.float32), RATE)
    runner = _runner(store, _Host())
    job_id = _submit_path(store, runner, empty)

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "failed" and "no audio" in manifest["error"]


async def test_an_upload_that_never_completed_fails_cleanly(store):
    runner = _runner(store, _Host())
    job_id = store.create(params=PARAMS, source={"kind": "upload", "path": None})["id"]  # crashed mid-upload
    runner.submit(job_id)

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "failed" and manifest["error"] == "The upload did not complete"


# --- model availability and hot-swap -------------------------------------------


async def test_a_job_waits_for_the_model_instead_of_failing(store, media):
    host = _Host()
    host.loaded = False
    runner = _runner(store, host)
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        await _wait_for(lambda: store.load(job_id)["phase"] == "waiting_for_model")
        await asyncio.sleep(0.1)
        assert store.load(job_id)["status"] == "running"
        assert host.calls == []

        host.loaded = True
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "completed"
    assert host.offsets == OFFSETS


async def test_a_job_can_be_cancelled_while_waiting_for_the_model(store, media):
    host = _Host()
    host.loaded = False
    runner = _runner(store, host)
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        await _wait_for(lambda: store.load(job_id)["phase"] == "waiting_for_model")
        store.request_cancel(job_id)
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "cancelled"
    assert host.calls == []


async def test_a_swapped_model_is_picked_up_between_chunks(store, media):
    old, new = _Host(), _Host()
    current = [old]
    old.hook = lambda offset: offset == 10.0 and current.__setitem__(0, new)
    runner = JobRunner(
        store,
        lambda: current[0],
        InferenceQueue(),
        RunnerConfig(retry_backoff=(0.0, 0.0), model_poll_seconds=0.01),
        normalize=_copy_normalize,
        find_gap=_no_gap,
    )
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        await _finish(store, runner, job_id, "completed")

    assert old.offsets == [0.0, 5.0, 10.0]
    assert new.offsets == [15.0, 20.0, 25.0]


# --- sharing the model with live captions ------------------------------------


async def test_a_live_call_is_served_before_the_next_batch_chunk(store, media):
    queue = InferenceQueue()
    order = []
    release = threading.Event()
    host = _Host()

    def hook(offset):
        order.append(f"chunk {offset:g} start")
        if offset == 10.0:
            release.wait(10)
        order.append(f"chunk {offset:g} end")

    host.hook = hook
    runner = _runner(store, host, queue)
    job_id = _submit_path(store, runner, media)

    try:
        async with _running(runner):
            await _wait_for(lambda: "chunk 10 start" in order)
            live = asyncio.create_task(queue.run(lambda: order.append("live")))
            await _wait_for(lambda: queue.stats()["live_depth"] == 1)  # the live call waits behind the running chunk
            release.set()
            await live
            await _finish(store, runner, job_id, "completed")
    finally:
        release.set()

    position = order.index("chunk 10 end")
    assert order[position : position + 3] == ["chunk 10 end", "live", "chunk 15 start"]


# --- housekeeping ----------------------------------------------------------------


async def test_finished_jobs_are_purged_after_the_ttl(store, media):
    runner = _runner(store, _Host(), ttl_hours=0, purge_interval_seconds=0.01)
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        await _wait_for(lambda: _status(store, job_id) is None)  # completed, then purged by the sweep

    assert store.list() == []


async def test_finished_jobs_are_kept_within_the_ttl(store, media):
    runner = _runner(store, _Host(), ttl_hours=72, purge_interval_seconds=0.01)
    job_id = _submit_path(store, runner, media)

    async with _running(runner):
        await _finish(store, runner, job_id, "completed")
        await asyncio.sleep(0.1)  # several sweeps

    assert store.load(job_id)["status"] == "completed"


async def test_a_cancel_during_the_retry_backoff_stops_the_job_instead_of_retrying(store, media):
    host = _Host()
    runner = _runner(store, host)
    runner.config.retry_backoff = (0.2, 0.2)
    job_id = _submit_path(store, runner, media)

    def fail_once(offset):
        if offset == 10.0:
            store.request_cancel(job_id)  # the client deletes the job while the failing chunk backs off
            raise RuntimeError("out of memory")

    host.hook = fail_once

    async with _running(runner):
        manifest = await _finish(store, runner, job_id)

    assert manifest["status"] == "cancelled"
    assert host.offsets == [0.0, 5.0, 10.0]  # no second attempt at chunk 2
