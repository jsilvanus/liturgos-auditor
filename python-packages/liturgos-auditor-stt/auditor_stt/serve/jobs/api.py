"""HTTP API for batch jobs, mounted under /v1/jobs.

    POST   /v1/jobs               submit a file (upload, a path on the media volume, or a URL to fetch)
    GET    /v1/jobs               list, filterable by client_ref / status
    GET    /v1/jobs/{id}          status and progress
    GET    /v1/jobs/{id}/result   the transcript as json, vtt, srt, text or youtube
    DELETE /v1/jobs/{id}          cancel and purge

The router is included behind the API key and answers 503 when the service has
no data directory. Handlers only read and write the job store; the runner does
the work. Oversized or unauthenticated uploads are refused before the body is
read by serve/limits.py, because FastAPI parses the body first.
"""

import asyncio
import fnmatch
import re
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

import httpx

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.convertors import Convertor, register_url_convertor

from ...captions import make_cues, parse_start_time, to_srt, to_text, to_vtt, to_youtube
from .assemble import IncompleteJobError, assemble, progress
from .chunking import MAX_CHUNK_SECONDS, MIN_CHUNK_SECONDS
from .store import STATUSES, JobNotFoundError

JOBS_NOT_CONFIGURED = "Jobs are not configured (set AUDITOR_STT_DATA_DIR)"

FORMATS = ("json", "vtt", "srt", "text", "youtube")
MEDIA_TYPES = {
    "vtt": "text/vtt; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "text": "text/plain; charset=utf-8",
    "youtube": "text/plain; charset=utf-8",
}

_RESERVED_NAMES = re.compile(r"(?i)(con|prn|aux|nul|com\d|lpt\d)(\.|$)")


class _JobIdConvertor(Convertor):
    """Only uuid4 hex strings match `{job_id:jobid}`, so /v1/jobs/<anything else> is not swallowed by
    these routes and stays free for routers included after this one."""

    regex = "[0-9a-f]{32}"

    def convert(self, value):
        return value

    def to_string(self, value):
        return value


register_url_convertor("jobid", _JobIdConvertor())


def _require_jobs(request: Request):
    if request.app.state.job_store is None:
        raise HTTPException(status_code=503, detail=JOBS_NOT_CONFIGURED)


router = APIRouter(prefix="/v1/jobs", dependencies=[Depends(_require_jobs)])


# --- helpers ---------------------------------------------------------------


def _load(store, job_id):
    try:
        return store.load(job_id)
    except JobNotFoundError:  # includes an id that is not even well formed
        raise HTTPException(status_code=404, detail="No such job") from None


def _job_view(manifest, runner):
    plan = manifest.get("chunks") or []
    current = manifest.get("current_seconds") or 0.0
    # The runner works in order, so the chunks done are those ending at or before the current position.
    done = sum(1 for chunk in plan if chunk["end"] <= current + 1e-6)
    baseline = runner.baseline_chunks(manifest["id"]) if runner is not None else 0
    numbers = progress(manifest, done, baseline_chunks=baseline)

    status = manifest["status"]
    eta = None if status in ("failed", "cancelled") else numbers["eta_seconds"]
    params = manifest.get("params") or {}
    return {
        "id": manifest["id"],
        "status": status,
        "phase": manifest.get("phase"),
        "error": manifest.get("error"),
        "client_ref": params.get("client_ref"),
        "created_at": manifest.get("created_at"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "progress": round(numbers["progress"], 1),
        "current_seconds": round(numbers["current_seconds"], 1),
        "total_seconds": round(numbers["total_seconds"], 1),
        "eta_seconds": None if eta is None else round(eta, 1),
        "chunks_done": numbers["chunks_done"],
        "chunks_total": numbers["chunks_total"],
        "params": {
            "language": params.get("language"),
            "chunk_seconds": params.get("chunk_seconds"),
            "word_timestamps": params.get("word_timestamps"),
        },
        "cancel_requested": bool(manifest.get("cancel_requested")),
    }


def _bad_source():
    return HTTPException(status_code=422, detail="source_path must name an existing file inside the media root")


def _resolve_source(media_root, value):
    """The real path of a file under the media root, or 422.

    A path that is lexically elsewhere is refused before the filesystem is
    touched (a UNC path would make Windows contact that host). The rest is
    resolved first, so `..` segments and symlinks are followed before the
    containment check and neither can lead out of the root. Every rejection
    says the same thing: whether a file exists elsewhere is not disclosed.
    """
    if media_root is None:
        raise HTTPException(status_code=422, detail="source_path is not enabled (set AUDITOR_STT_MEDIA_ROOT)")
    try:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = media_root / candidate
        if not candidate.is_relative_to(media_root):
            raise _bad_source()
        resolved = candidate.resolve(strict=True)
    except (OSError, ValueError, RuntimeError):  # missing, malformed, embedded NUL, symlink loop
        raise _bad_source() from None
    if not resolved.is_file() or not resolved.is_relative_to(media_root):
        raise _bad_source()
    return resolved


def _safe_name(filename):
    """A file name that is safe to create inside the job's source directory."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path((filename or "").replace("\\", "/")).name).strip("._")
    name = name[-100:] or "upload"
    return f"_{name}" if _RESERVED_NAMES.match(name) else name


class _UploadTooLargeError(Exception):
    pass


class _CappedReader:
    """File wrapper for copyfileobj that fails once more than `limit` bytes were read."""

    def __init__(self, file, limit):
        self._file = file
        self._left = limit

    def read(self, size=-1):
        data = self._file.read(size)
        self._left -= len(data)
        if self._left < 0:
            raise _UploadTooLargeError()
        return data


def _save_upload(file, target, limit):
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as out:
        shutil.copyfileobj(_CappedReader(file, limit), out, 1024 * 1024)
    return target.stat().st_size


def _bad_url(detail="source_url could not be used"):
    return HTTPException(status_code=422, detail=detail)


def _check_source_url(allowed_hosts, value):
    """The URL, if `AUDITOR_STT_SOURCE_URL_HOSTS` allows its host; 422 otherwise.

    The service fetches the URL itself, so the operator's host list is the only thing keeping this
    from reaching arbitrary addresses. Only http(s) URLs without credentials are accepted, and a
    host that is not listed gets the same answer as any other unusable URL.
    """
    if not allowed_hosts:
        raise _bad_url("source_url is not enabled (set AUDITOR_STT_SOURCE_URL_HOSTS)")
    try:
        parts = urlsplit(value.strip())
        host = (parts.hostname or "").lower()
        port = parts.port  # raises ValueError for a malformed port
    except ValueError:
        raise _bad_url() from None
    if parts.scheme not in ("http", "https") or not host or parts.username or parts.password:
        raise _bad_url()
    if not any(fnmatch.fnmatchcase(host, pattern) for pattern in allowed_hosts):
        raise _bad_url()
    return parts.geturl(), (unquote(Path(parts.path).name) if parts.path else "")


class _FetchError(Exception):
    pass


def _fetch_url(url, target, limit, timeout):
    """Stream `url` into `target`; no redirects are followed. Returns the size in bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with httpx.stream("GET", url, follow_redirects=False, timeout=timeout) as response:
            if response.status_code != 200:
                raise _FetchError(f"source_url answered {response.status_code}")
            with open(target, "wb") as out:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise _UploadTooLargeError()
                    out.write(chunk)
    except httpx.HTTPError:
        raise _FetchError("source_url could not be fetched") from None
    return size


def _repair_offset(value):
    # An unencoded "+02:00" in a query string arrives as " 02:00".
    return re.sub(r" (\d\d:\d\d)$", r"+\1", value.strip())


# --- routes ----------------------------------------------------------------


@router.post("", status_code=202)
async def submit_job(
    request: Request,
    file: Optional[UploadFile] = File(None),
    source_path: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    chunk_seconds: float = Form(60.0, ge=MIN_CHUNK_SECONDS, le=MAX_CHUNK_SECONDS),
    word_timestamps: bool = Form(True),
    prompt: Optional[str] = Form(None),
    client_ref: Optional[str] = Form(None, max_length=200),
):
    state = request.app.state
    store, runner = state.job_store, state.job_runner
    if (file is not None) + bool(source_path) + bool(source_url) != 1:
        raise HTTPException(status_code=422, detail="Provide exactly one of 'file', 'source_path' or 'source_url'")

    resolved = None
    fetch = None
    if source_url:
        fetch = _check_source_url(state.source_url_hosts, source_url)
    if source_path:
        resolved = await asyncio.to_thread(_resolve_source, state.media_root, source_path)

    params = {
        "language": language or state.default_language,
        "chunk_seconds": chunk_seconds,
        "word_timestamps": word_timestamps,
        "prompt": prompt or None,
        "client_ref": client_ref or None,
    }
    if resolved is not None:
        manifest = await asyncio.to_thread(store.create, params=params, source={"kind": "path", "path": resolved})
    elif fetch is not None:
        manifest = await _create_from_url(state, params, *fetch)
    else:
        manifest = await _create_from_upload(state, params, file)

    runner.submit(manifest["id"])
    return JSONResponse(
        {"id": manifest["id"], "status": "queued"},
        status_code=202,
        headers={"Location": f"/v1/jobs/{manifest['id']}"},
    )


async def _create_from_upload(state, params, file):
    store = state.job_store
    manifest = await asyncio.to_thread(store.create, params=params, source={"kind": "upload", "path": None})
    job_id = manifest["id"]
    target = store.source_dir(job_id) / _safe_name(file.filename)
    try:
        size = await asyncio.to_thread(_save_upload, file.file, target, state.max_upload_bytes)
        if size == 0:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")
        # Recorded only now: a job whose upload never finished has no source and fails cleanly.
        return await asyncio.to_thread(store.update, job_id, source={"kind": "upload", "path": str(target)})
    except _UploadTooLargeError:
        await asyncio.to_thread(store.delete, job_id)
        limit_mb = state.max_upload_bytes / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Upload is too large (limit is {limit_mb:g} MB)") from None
    except BaseException:
        await asyncio.to_thread(store.delete, job_id)  # never leave a partial upload behind
        raise


async def _create_from_url(state, params, url, name):
    """Like an upload: the file is fetched into the job's source directory and removed when the job ends."""
    store = state.job_store
    manifest = await asyncio.to_thread(store.create, params=params, source={"kind": "upload", "path": None})
    job_id = manifest["id"]
    target = store.source_dir(job_id) / _safe_name(name or "source")
    try:
        size = await asyncio.to_thread(_fetch_url, url, target, state.max_upload_bytes, state.source_url_timeout)
        if size == 0:
            raise HTTPException(status_code=422, detail="The fetched file is empty")
        return await asyncio.to_thread(store.update, job_id, source={"kind": "upload", "path": str(target)})
    except _UploadTooLargeError:
        await asyncio.to_thread(store.delete, job_id)
        limit_mb = state.max_upload_bytes / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"The file is too large (limit is {limit_mb:g} MB)") from None
    except _FetchError as exc:
        await asyncio.to_thread(store.delete, job_id)
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except BaseException:
        await asyncio.to_thread(store.delete, job_id)
        raise


@router.get("")
def list_jobs(request: Request, client_ref: Optional[str] = None, status: Optional[str] = None):
    if status is not None and status not in STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of: {', '.join(STATUSES)}")
    state = request.app.state
    manifests = state.job_store.list()  # oldest first
    jobs = [
        _job_view(manifest, state.job_runner)
        for manifest in reversed(manifests)
        if (client_ref is None or (manifest.get("params") or {}).get("client_ref") == client_ref)
        and (status is None or manifest.get("status") == status)
    ]
    return {"jobs": jobs}


@router.get("/{job_id:jobid}")
def get_job(job_id: str, request: Request):
    state = request.app.state
    return _job_view(_load(state.job_store, job_id), state.job_runner)


@router.get("/{job_id:jobid}/result")
def get_result(
    job_id: str,
    request: Request,
    fmt: str = Query("json", alias="format"),
    partial: bool = False,
    start_time: Optional[str] = None,
    region: Optional[str] = None,
    cue: Optional[str] = None,
    max_cue_duration: float = Query(7.0, gt=0),
    max_line_chars: int = Query(42, ge=1),
):
    if fmt not in FORMATS:
        raise HTTPException(status_code=422, detail=f"Unsupported format '{fmt}' (supported: {', '.join(FORMATS)})")
    start = None
    if fmt == "youtube":
        if not start_time:
            raise HTTPException(
                status_code=422, detail="format=youtube requires start_time (ISO-8601 UTC, media time 0)"
            )
        try:
            start = parse_start_time(_repair_offset(start_time))
        except ValueError:
            raise HTTPException(status_code=422, detail="start_time is not a valid ISO-8601 timestamp") from None

    store = request.app.state.job_store
    manifest = _load(store, job_id)
    if manifest["status"] != "completed" and not partial:
        # Failed and cancelled jobs are readable too, but only by asking for the partial result.
        return JSONResponse({"detail": "Job is not finished", "status": manifest["status"]}, status_code=409)

    total = len(manifest.get("chunks") or [])
    chunk_results = {index: store.read_chunk(job_id, index) for index in range(total)}
    try:
        result = assemble(manifest, chunk_results, partial=partial)
    except IncompleteJobError:
        raise HTTPException(status_code=500, detail="Result files of this job are missing") from None

    if fmt == "json":
        return JSONResponse(result)

    if fmt == "text":
        body = to_text(result)
    else:
        cues = make_cues(result, max_cue_duration, max_line_chars)
        if fmt == "vtt":
            body = to_vtt(cues)
        elif fmt == "srt":
            body = to_srt(cues)
        else:
            try:
                body = to_youtube(cues, start, region=region, cue=cue)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
    return Response(
        body, media_type=MEDIA_TYPES[fmt], headers={"X-Job-Complete": "true" if result["complete"] else "false"}
    )


@router.delete("/{job_id:jobid}")
async def delete_job(job_id: str, request: Request):
    state = request.app.state
    store, runner = state.job_store, state.job_runner
    manifest = await asyncio.to_thread(_load, store, job_id)

    try:
        if manifest["status"] == "running":
            await asyncio.to_thread(store.request_cancel, job_id)
            # The runner still holds the audio open; it deletes the job once it has stopped.
            if runner.discard(job_id):
                return JSONResponse({"id": job_id, "status": "running", "cancel_requested": True}, status_code=202)
        await asyncio.to_thread(store.delete, job_id)
    except JobNotFoundError:
        pass  # removed by someone else in the meantime: the outcome asked for
    except OSError:
        # Windows will not delete files another handle has open, e.g. just as the runner finishes.
        raise HTTPException(status_code=409, detail="Job is busy; try again") from None
    return Response(status_code=204)
