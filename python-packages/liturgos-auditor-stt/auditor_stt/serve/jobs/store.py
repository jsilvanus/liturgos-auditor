"""Filesystem job store: one directory per job, JSON for everything.

Layout under the store root:

    <job_id>/manifest.json        job state (schema below)
    <job_id>/chunks/00042.json    one finished chunk result, absolute timestamps
    <job_id>/source/              uploaded input, if any (created by the caller)
    <job_id>/audio.wav            normalised audio (created by the runner)

Every JSON file is written to `<name>.tmp`, fsynced and renamed over the
target, so a crash leaves either the old or the new content, never a torn
file, and a chunk file that exists is a finished chunk. Read-modify-write of
a manifest happens under one lock, which is enough for a single-process
service; several processes sharing a store are not supported.

Job ids are uuid4 hex and validated on every call, so an id can never
address a path outside the root.

Manifest (times are ISO-8601 UTC strings, null until known):

    id                str
    status            queued | running | completed | failed | cancelled
    created_at, updated_at, started_at, finished_at
                      started_at is when the CURRENT run began (reset on
                      every transition to running, cleared on requeue)
    params            {language, chunk_seconds, word_timestamps, prompt, client_ref}
    source            {kind: "upload" | "path" | "url", path, url?, ingest_token?}
    audio             {wav_path, duration_seconds, sample_rate} | null
    chunks            [{index, start_sample, end_sample, start, end, snapped}] | null
                      the persisted plan; never recomputed on resume
    current_seconds   audio seconds transcribed so far
    phase             free-form label for the runner, e.g. "transcribing"
    error             str | null
    cancel_requested  bool
"""

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATUSES = ("queued", "running", "completed", "failed", "cancelled")
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
SOURCE_KINDS = ("upload", "path", "url")

DEFAULT_PARAMS = {
    "language": "fi",
    "chunk_seconds": 60.0,
    "word_timestamps": True,
    "prompt": None,
    "client_ref": None,
}

# Fields update() may change. Status and its timestamps go through set_status().
_UPDATABLE = frozenset(
    {"params", "source", "audio", "chunks", "current_seconds", "phase", "error", "cancel_requested"}
)

_ID_RE = re.compile(r"[0-9a-f]{32}")
_CHUNK_FILE_RE = re.compile(r"(\d+)\.json")


class JobNotFoundError(LookupError):
    pass


class InvalidJobIdError(JobNotFoundError, ValueError):
    """The id is not a uuid4 hex string; a job with such an id cannot exist."""


def _utcnow():
    return datetime.now(timezone.utc)


def _iso(moment):
    # Fixed width, so the strings also sort chronologically.
    return moment.isoformat(timespec="microseconds")


def _replace(source, target):
    # Windows refuses to replace a file another handle has open (an antivirus
    # scan or indexer right after we wrote it); that clears within moments.
    for attempt in range(5):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02 * (attempt + 1))


def _atomic_write_json(path, data):
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class JobStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Reentrant so purge_expired/recover can call the locking public methods.
        # Manifest reads take it too: on Windows a rename over a file that is
        # being read fails.
        self._lock = threading.RLock()

    # --- paths ---------------------------------------------------------

    def job_dir(self, job_id):
        if not isinstance(job_id, str) or not _ID_RE.fullmatch(job_id):
            raise InvalidJobIdError("Invalid job id")
        return self.root / job_id

    def source_dir(self, job_id):
        """Where an uploaded input lives. Not created here."""
        return self.job_dir(job_id) / "source"

    def audio_path(self, job_id):
        """Where the normalised WAV goes. Not created here."""
        return self.job_dir(job_id) / "audio.wav"

    def _manifest_path(self, job_id):
        return self.job_dir(job_id) / "manifest.json"

    def _chunk_path(self, job_id, index):
        if not isinstance(index, int) or index < 0:
            raise ValueError("Chunk index must be a non-negative integer")
        return self.job_dir(job_id) / "chunks" / f"{index:05d}.json"

    # --- manifest ------------------------------------------------------

    def _read_manifest(self, job_id):
        try:
            manifest = _load_json(self._manifest_path(job_id))
        except FileNotFoundError:
            raise JobNotFoundError(f"No such job: {job_id}") from None
        if not isinstance(manifest, dict):
            raise ValueError(f"Manifest of job {job_id} is not an object")
        return manifest

    def _modify(self, job_id, mutate):
        with self._lock:
            manifest = self._read_manifest(job_id)
            mutate(manifest)
            manifest["updated_at"] = _iso(_utcnow())
            _atomic_write_json(self._manifest_path(job_id), manifest)
            return manifest

    def create(self, *, params, source, job_id=None):
        """Create a queued job and return its manifest.

        `params` is filled up from DEFAULT_PARAMS. `source["path"]` may be
        None here and set later with update(source=...), for uploads that are
        written into source_dir() after the job exists.
        """
        directory = self.job_dir(uuid.uuid4().hex if job_id is None else job_id)
        job_id = directory.name
        if source.get("kind") not in SOURCE_KINDS:
            raise ValueError(f"source kind must be one of {SOURCE_KINDS}")

        now = _iso(_utcnow())
        manifest = {
            "id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "params": {**DEFAULT_PARAMS, **params},
            "source": {
                "kind": source["kind"],
                "path": None if source.get("path") is None else str(source["path"]),
                # kind "url": the file is stripped elsewhere and arrives as audio.wav (see fleet.py)
                **{key: source[key] for key in ("url", "ingest_token") if source.get(key)},
            },
            "audio": None,
            "chunks": None,
            "current_seconds": 0.0,
            "phase": "queued",
            "error": None,
            "cancel_requested": False,
        }
        with self._lock:
            (directory / "chunks").mkdir(parents=True)  # FileExistsError if the id is taken
            try:
                _atomic_write_json(directory / "manifest.json", manifest)
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
        return manifest

    def load(self, job_id):
        with self._lock:
            return self._read_manifest(job_id)

    def update(self, job_id, **fields):
        """Merge top-level manifest fields (nested values are replaced whole)."""
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"Cannot update manifest field(s): {', '.join(sorted(unknown))}")
        return self._modify(job_id, lambda manifest: manifest.update(fields))

    def set_status(self, job_id, status, error=None):
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")

        def mutate(manifest):
            now = _iso(_utcnow())
            manifest["status"] = status
            manifest["error"] = error
            if status == "running":
                manifest["started_at"] = now
                manifest["finished_at"] = None
            elif status == "queued":
                manifest["started_at"] = None
                manifest["finished_at"] = None
            else:
                manifest["finished_at"] = now

        return self._modify(job_id, mutate)

    def _scan(self):
        """All readable manifests, oldest first. Damaged jobs are skipped, not fatal."""
        manifests = []
        for entry in os.scandir(self.root):
            if not entry.is_dir() or not _ID_RE.fullmatch(entry.name):
                continue
            try:
                manifests.append(self._read_manifest(entry.name))
            except (JobNotFoundError, OSError, ValueError):
                logger.warning("Skipping job %s: manifest is missing or unreadable", entry.name)
        manifests.sort(key=lambda m: (m.get("created_at") or "", m.get("id") or ""))
        return manifests

    def list(self):
        """Every job's manifest, oldest first."""
        with self._lock:
            return self._scan()

    # --- cancellation --------------------------------------------------

    def request_cancel(self, job_id):
        return self._modify(job_id, lambda manifest: manifest.update(cancel_requested=True))

    def cancel_requested(self, job_id):
        """True once cancellation was requested. A deleted job counts as cancelled.

        DELETE removes the directory while the runner may still be mid-chunk;
        for the runner both mean "stop".
        """
        try:
            return bool(self.load(job_id).get("cancel_requested"))
        except InvalidJobIdError:
            raise
        except JobNotFoundError:
            return True

    # --- chunk results -------------------------------------------------

    def write_chunk(self, job_id, index, result):
        path = self._chunk_path(job_id, index)
        if not path.parent.is_dir():
            raise JobNotFoundError(f"No such job: {job_id}")
        _atomic_write_json(path, result)

    def read_chunk(self, job_id, index):
        """The stored result, or None if the chunk is missing or unreadable."""
        path = self._chunk_path(job_id, index)  # validates outside the try: a bad id is an error, not "missing"
        try:
            result = _load_json(path)
        except (OSError, ValueError):
            return None
        return result if isinstance(result, dict) else None

    def done_indices(self, job_id):
        """Sorted indices of chunks with a readable result file.

        A corrupt file counts as not done, so resume simply redoes it. This
        parses every chunk file: use it at resume time, not on each poll.
        """
        chunks_dir = self.job_dir(job_id) / "chunks"
        try:
            entries = list(os.scandir(chunks_dir))
        except FileNotFoundError:
            raise JobNotFoundError(f"No such job: {job_id}") from None

        done = []
        for entry in entries:
            match = _CHUNK_FILE_RE.fullmatch(entry.name)
            if not match:
                continue  # includes leftover ".json.tmp" files
            try:
                if isinstance(_load_json(entry.path), dict):
                    done.append(int(match.group(1)))
            except (OSError, ValueError):
                continue
        return sorted(done)

    # --- lifecycle -----------------------------------------------------

    def delete(self, job_id):
        """Remove the job and everything in it. Returns False if there was no such job."""
        with self._lock:
            directory = self.job_dir(job_id)
            if not directory.exists():
                return False
            shutil.rmtree(directory)
            return True

    def remove_working_files(self, job_id):
        """Delete the uploaded source and the normalised WAV, keeping manifest and results.

        Only touches this job's own source/ directory and audio.wav; a
        `source.kind == "path"` input lives elsewhere and is never removed.
        The caller must have closed the WAV first (Windows).
        """
        source = self.source_dir(job_id)
        if source.is_dir():
            shutil.rmtree(source)
        self.audio_path(job_id).unlink(missing_ok=True)

    def purge_expired(self, ttl_hours, now=None):
        """Delete finished jobs whose end is more than `ttl_hours` before `now`; returns their ids.

        `now` is a timezone-aware datetime (default: the current time). Queued
        and running jobs are never purged, however old.
        """
        if ttl_hours < 0:
            raise ValueError("ttl_hours must not be negative")
        cutoff = (now or _utcnow()) - timedelta(hours=ttl_hours)
        purged = []
        with self._lock:
            for manifest in self._scan():
                if manifest.get("status") not in TERMINAL_STATUSES:
                    continue
                ended = manifest.get("finished_at") or manifest.get("updated_at")
                try:
                    expired = datetime.fromisoformat(ended) < cutoff
                except (TypeError, ValueError):
                    continue
                if expired and self.delete(manifest["id"]):
                    purged.append(manifest["id"])
        return purged

    def recover(self):
        """Crash recovery at startup: running jobs go back to queued.

        Returns the ids of all queued jobs, oldest first. Finished chunk
        files are untouched, so the runner resumes from them.
        """
        with self._lock:
            for manifest in self._scan():
                if manifest.get("status") == "running":
                    self.set_status(manifest["id"], "queued")
                    self.update(manifest["id"], phase="queued")
            return [m["id"] for m in self._scan() if m.get("status") == "queued"]
