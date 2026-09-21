"""Runs batch jobs: one at a time, chunk by chunk, each chunk checkpointed to disk.

A single worker takes queued jobs in FIFO order. Per job it normalises the
media to a WAV, plans the chunks once (persisted in the manifest), then
transcribes them in order through the inference queue at BATCH priority - a
live caption request that arrives meanwhile is served before the next chunk
starts. Every finished chunk is written atomically before the next one begins,
so a crash or a graceful stop loses at most the chunk in flight, and a job
found `running` at startup simply resumes from the chunk files on disk.

The host is fetched per chunk, so a model switch (POST /model) or a restart
with another model in the middle of a job continues it on the new model: the
job's chunks then come from different models. Nothing prevents or repairs
that; each chunk result records its `model` and the assembled result lists
every model used as `models`, so a mixed transcript is at least visible.

Nothing here logs audio or transcript text: job ids, chunk indexes, timings
and error class/messages only. A source of kind "path" lives on the caller's
media volume and is never deleted; only the job's own uploaded copy and its
normalised WAV are.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass

from ..audio import MediaDecodeError, NormalizationCancelledError, normalize_to_wav
from .chunking import Chunk
from .pipeline import plan_job_chunks, transcribe_chunk
from .store import TERMINAL_STATUSES, JobNotFoundError
from .wav import open_pcm16_mono

logger = logging.getLogger(__name__)

MAX_CHUNK_RETRIES = 2
_ERROR_LIMIT = 300  # characters of an exception message kept in the manifest


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


@dataclass
class RunnerConfig:
    ttl_hours: float = 72.0  # how long finished jobs (and their results) are kept
    carry_chars: int = 200  # tail of the previous chunk's text used as prompt; 0 disables
    batch_vad: bool = True  # Silero VAD inside each chunk; measured against unchunked runs
    # Timing knobs: tests shrink them, deployments have no reason to change them.
    retry_backoff: tuple = (1.0, 4.0)  # seconds before the 1st and 2nd retry of a chunk
    model_poll_seconds: float = 1.0
    purge_interval_seconds: float = 3600.0

    @classmethod
    def from_env(cls):
        return cls(
            ttl_hours=float(os.environ.get("AUDITOR_STT_JOB_TTL_HOURS", cls.ttl_hours)),
            carry_chars=int(os.environ.get("AUDITOR_STT_CARRY_CONTEXT_CHARS", cls.carry_chars)),
            batch_vad=_env_bool("AUDITOR_STT_BATCH_VAD", cls.batch_vad),
        )


class _JobCancelled(Exception):
    pass


class _JobFailed(Exception):
    """The job cannot go on; the message is what the client sees as `error`."""


class _Run:
    """What one job in flight owns."""

    def __init__(self, job_id):
        self.job_id = job_id
        self.wav = None
        self.end_seconds = 0.0
        # Set to interrupt a running ffmpeg (stop, or DELETE while normalising).
        self.cancel = threading.Event()

    def close(self):
        wav, self.wav = self.wav, None
        if wav is not None:
            wav.close()


class JobRunner:
    """`get_host()` is called before every chunk, so a model hot-swap is picked up between chunks."""

    def __init__(self, store, get_host, queue, config=None, *, normalize=normalize_to_wav, find_gap=None):
        self.store = store
        self.config = config or RunnerConfig.from_env()
        self._get_host = get_host
        self._queue = queue
        self._normalize = normalize
        self._find_gap = find_gap
        self._pending = asyncio.Queue()
        self._worker = None
        self._purger = None
        self._run = None
        self._discard = set()
        self._baseline = {}

    # --- lifecycle -------------------------------------------------------

    async def start(self):
        """Recover from a previous run, requeue what was unfinished, and start working."""
        for job_id in await asyncio.to_thread(self._recover):
            self._pending.put_nowait(job_id)
        self._worker = asyncio.create_task(self._work(), name="job-runner")
        self._purger = asyncio.create_task(self._purge_periodically(), name="job-purger")

    async def stop(self):
        """Stop promptly: an in-flight chunk is abandoned, not awaited.

        The job stays `running` in its manifest, so the next start() resumes it
        from the chunk files. A model call cannot be interrupted, so its thread
        may still finish in the background.
        """
        if self._run is not None:
            self._run.cancel.set()
        tasks = [task for task in (self._worker, self._purger) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._worker = self._purger = None

    def submit(self, job_id):
        self._pending.put_nowait(job_id)

    def discard(self, job_id):
        """Cancel a running job and delete it once it has stopped (DELETE while running).

        Returns False if the runner is not working on that job; the caller then
        deletes it directly. Call from the event loop.
        """
        run = self._run
        if run is None or run.job_id != job_id:
            return False
        self._discard.add(job_id)
        run.cancel.set()
        return True

    def baseline_chunks(self, job_id):
        """Chunks already done when this run of the job began (for ETA after a resume)."""
        return self._baseline.get(job_id, 0)

    def _recover(self):
        queued = self.store.recover()
        # A crash between "job finished" and "working files removed" would leave audio behind.
        for manifest in self.store.list():
            if manifest.get("status") in TERMINAL_STATUSES:
                try:
                    self.store.remove_working_files(manifest["id"])
                except OSError:
                    logger.warning("Job %s: could not remove leftover working files", manifest["id"])
        return queued

    async def _purge_periodically(self):
        while True:
            try:
                purged = await asyncio.to_thread(self.store.purge_expired, self.config.ttl_hours)
                if purged:
                    logger.info("Purged %d expired job(s)", len(purged))
            except Exception:  # noqa: BLE001 - a failed sweep must not end the loop
                logger.exception("Purging expired jobs failed")
            await asyncio.sleep(self.config.purge_interval_seconds)

    # --- the worker ------------------------------------------------------

    async def _work(self):
        while True:
            job_id = await self._pending.get()
            try:
                await self._run_job(job_id)
            except Exception:  # noqa: BLE001 - one broken job must not stop the worker
                logger.exception("Job %s: runner error", job_id)
                self._run = None

    async def _run_job(self, job_id):
        try:
            manifest = await asyncio.to_thread(self.store.load, job_id)
        except JobNotFoundError:
            return  # deleted while queued
        if manifest.get("status") != "queued":
            return  # a duplicate submit, or already handled

        run = self._run = _Run(job_id)
        outcome = None  # (status, error); None when the job vanished
        try:
            await self._execute(run, manifest)
            outcome = ("completed", None)
        except _JobCancelled:
            outcome = ("cancelled", None)
        except _JobFailed as exc:
            outcome = ("failed", str(exc))
        except JobNotFoundError:
            logger.info("Job %s: removed while running", job_id)
        except asyncio.CancelledError:
            # stop(): the manifest still says running, and start() resumes it. Keep every file.
            run.close()
            self._run = None
            raise
        except Exception as exc:  # noqa: BLE001 - whatever it was, it ends this job and no other
            logger.exception("Job %s: unexpected error", job_id)
            outcome = ("failed", _describe(exc))
        await self._finish(run, outcome)

    async def _finish(self, run, outcome):
        job_id = run.job_id
        run.close()  # Windows cannot delete a file that is still mapped
        gone = outcome is None
        if outcome is not None:
            status, error = outcome
            try:
                await asyncio.to_thread(self._finalise, run, status, error)
                logger.info("Job %s: %s", job_id, status if error is None else f"{status} ({error})")
            except JobNotFoundError:
                gone = True
            except OSError:
                logger.exception("Job %s: could not record the final state", job_id)

        # No await between reading the discard mark and clearing the run: a discard() that
        # arrives later sees no run for this job and its caller deletes the job itself.
        discard = job_id in self._discard
        self._discard.discard(job_id)
        self._baseline.pop(job_id, None)
        self._run = None
        if discard or gone:
            await asyncio.to_thread(self._delete_quietly, job_id)

    def _finalise(self, run, status, error):
        fields = {"phase": status}
        if status == "completed":
            fields["current_seconds"] = run.end_seconds
        self.store.set_status(run.job_id, status, error=error)
        self.store.update(run.job_id, **fields)
        self.store.remove_working_files(run.job_id)

    def _delete_quietly(self, job_id):
        try:
            self.store.delete(job_id)
        except OSError:
            logger.warning("Job %s: could not delete the job directory", job_id)

    # --- one job ---------------------------------------------------------

    async def _execute(self, run, manifest):
        store, job_id = self.store, run.job_id
        params = manifest["params"]
        await asyncio.to_thread(self._begin, job_id)
        await self._check_cancelled(run)

        wav_path = store.audio_path(job_id)
        if not (manifest.get("audio") and wav_path.is_file()):
            await self._normalise(run, manifest, wav_path)
            await self._check_cancelled(run)
        try:
            run.wav = open_pcm16_mono(wav_path)
        except (OSError, ValueError) as exc:
            raise _JobFailed(f"Normalised audio is unreadable: {_describe(exc)}") from exc

        manifest = await asyncio.to_thread(store.load, job_id)
        if not manifest.get("audio"):
            audio = {
                "wav_path": str(wav_path),
                "duration_seconds": run.wav.duration,
                "sample_rate": run.wav.sample_rate,
            }
            await asyncio.to_thread(store.update, job_id, audio=audio)
        chunks = await self._plan(run, manifest, params)
        run.end_seconds = chunks[-1].end

        done = set(await asyncio.to_thread(store.done_indices, job_id))
        prefix = 0
        while prefix in done:
            prefix += 1
        self._baseline[job_id] = prefix
        await asyncio.to_thread(
            store.update, job_id, phase="transcribing", current_seconds=chunks[prefix - 1].end if prefix else 0.0
        )

        previous = None  # (index, text) of the last chunk handled in this run
        for chunk in chunks:
            if chunk.index in done:
                continue
            await self._check_cancelled(run)
            previous_text = await self._previous_text(job_id, chunk.index, previous)
            started = time.monotonic()
            result = await self._transcribe(run, chunk, params, previous_text)
            await asyncio.to_thread(self._save_chunk, job_id, chunk, result)
            previous = (chunk.index, result["text"])
            logger.info(
                "Job %s: chunk %d/%d done in %.1f s", job_id, chunk.index + 1, len(chunks), time.monotonic() - started
            )

    def _begin(self, job_id):
        self.store.set_status(job_id, "running")
        self.store.update(job_id, phase="normalising")

    async def _check_cancelled(self, run):
        if run.cancel.is_set() or await asyncio.to_thread(self.store.cancel_requested, run.job_id):
            raise _JobCancelled()

    async def _normalise(self, run, manifest, wav_path):
        source = manifest.get("source") or {}
        if not source.get("path"):
            # An upload is recorded in the manifest only once it was fully received.
            raise _JobFailed("The upload did not complete" if source.get("kind") == "upload" else "Job has no source")
        try:
            await asyncio.to_thread(self._normalize, source["path"], wav_path, cancel=run.cancel)
        except NormalizationCancelledError:
            raise _JobCancelled() from None
        except MediaDecodeError as exc:
            raise _JobFailed(str(exc)) from exc

    async def _plan(self, run, manifest, params):
        """The persisted chunk plan, made now if this is the job's first run."""
        if manifest.get("chunks") is not None:
            return [Chunk.from_dict(entry, run.wav.sample_rate) for entry in manifest["chunks"]]
        try:
            chunks = await asyncio.to_thread(plan_job_chunks, run.wav, params["chunk_seconds"], self._find_gap)
        except ValueError as exc:
            raise _JobFailed(f"Cannot plan chunks: {exc}") from exc
        if not chunks:
            raise _JobFailed("The media contains no audio")
        await asyncio.to_thread(self.store.update, run.job_id, chunks=[chunk.to_dict() for chunk in chunks])
        return chunks

    async def _previous_text(self, job_id, index, previous):
        if index == 0:
            return None
        if previous is not None and previous[0] == index - 1:
            return previous[1]
        result = await asyncio.to_thread(self.store.read_chunk, job_id, index - 1)  # resumed job
        return (result or {}).get("text")

    def _save_chunk(self, job_id, chunk, result):
        self.store.write_chunk(job_id, chunk.index, result)
        self.store.update(job_id, current_seconds=chunk.end)

    async def _transcribe(self, run, chunk, params, previous_text):
        failure = None
        for attempt in range(MAX_CHUNK_RETRIES + 1):
            if attempt:
                await self._check_cancelled(run)  # a DELETE during the backoff must not start another attempt
            host = await self._ready_host(run)
            try:
                return await self._queue.run_batch(
                    transcribe_chunk,
                    host,
                    run.wav,
                    chunk,
                    language=params["language"],
                    word_timestamps=params["word_timestamps"],
                    prompt=params.get("prompt"),
                    vad=self.config.batch_vad,
                    previous_text=previous_text,
                    carry_chars=self.config.carry_chars,
                )
            except Exception as exc:  # noqa: BLE001 - any model failure is retried the same way
                failure = exc
                logger.warning(
                    "Job %s: chunk %d attempt %d failed: %s", run.job_id, chunk.index, attempt + 1, _describe(exc)
                )
                if attempt < MAX_CHUNK_RETRIES:
                    await asyncio.sleep(self.config.retry_backoff[min(attempt, len(self.config.retry_backoff) - 1)])
        raise _JobFailed(f"Chunk {chunk.index} failed after {MAX_CHUNK_RETRIES + 1} attempts: {_describe(failure)}")

    async def _ready_host(self, run):
        """The current host once its model is loaded; waits (and honours cancel) until then."""
        waiting = False
        while True:
            host = self._get_host()
            if host is not None and host.loaded:
                if waiting:
                    await asyncio.to_thread(self.store.update, run.job_id, phase="transcribing")
                return host
            if not waiting:
                waiting = True
                logger.info("Job %s: waiting for the model to load", run.job_id)
                await asyncio.to_thread(self.store.update, run.job_id, phase="waiting_for_model")
            await self._check_cancelled(run)
            await asyncio.sleep(self.config.model_poll_seconds)


def _describe(exc):
    """Class and message of an exception, for `error` and the log; never transcript text."""
    message = str(exc).strip()
    text = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return text[:_ERROR_LIMIT]
