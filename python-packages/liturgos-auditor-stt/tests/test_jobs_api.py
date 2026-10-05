import asyncio
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.jobs.api import _safe_name
from auditor_stt.serve.jobs.runner import JobRunner, RunnerConfig
from auditor_stt.serve.jobs.store import JobStore
from auditor_stt.serve.jobs.wav import write_pcm16_wav
from auditor_stt.serve.queue import InferenceQueue

RATE = 16000
KEY = "s3cret-key"
FIVE = {"chunk_seconds": "5"}  # 5 s chunks: a 30 s file is six chunks starting at 0, 5, ..., 25
OFFSETS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
TERMINAL = ("completed", "failed", "cancelled")
NOT_CONFIGURED = "Jobs are not configured (set AUDITOR_STT_DATA_DIR)"


class _Host:
    model_id = "stub-model"
    device = "cpu"
    compute_type = "int8"
    loaded = True

    def __init__(self):
        self.calls = []
        self.hook = None  # hook(offset) runs in the model thread, before the result is returned

    def load(self):
        pass

    def transcribe(self, audio_path, language=None, **options):
        return {"text": "live", "language": language, "segments": []}

    def transcribe_array(self, samples, language=None, *, prompt=None, time_offset=0.0, **options):
        self.calls.append({"offset": time_offset, "prompt": prompt, "language": language, **options})
        if self.hook is not None:
            self.hook(time_offset)
        duration = len(samples) / RATE
        text = f"chunk at {time_offset:g}"
        word = {"start": time_offset, "end": time_offset + duration, "text": text, "probability": 0.9}
        segment = {"start": time_offset, "end": time_offset + duration, "text": text}
        return {"text": text, "language": language, "segments": [{**segment, "words": [word]}]}


def _no_gap(_samples, _rate):
    return None


def _copy_normalize(source, wav_path, *, cancel=None):
    shutil.copyfile(source, wav_path)


def _wav_bytes(tmp_path, seconds=30):
    path = tmp_path / "payload.wav"
    write_pcm16_wav(path, np.zeros(int(seconds * RATE), dtype=np.float32), RATE)
    return path.read_bytes()


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in (
        "AUDITOR_STT_DATA_DIR",
        "AUDITOR_STT_MEDIA_ROOT",
        "AUDITOR_STT_SOURCE_URL_HOSTS",
        "AUDITOR_STT_API_KEY",
        "AUDITOR_STT_MAX_UPLOAD_MB",
        "AUDITOR_STT_DEFAULT_LANGUAGE",
    ):
        monkeypatch.delenv(name, raising=False)


@contextmanager
def _service(tmp_path, *, host=None, api_key=None, media=True, normalize=_copy_normalize, upload_cap=None):
    host = host or _Host()
    queue = InferenceQueue()
    store = JobStore(tmp_path / "data" / "jobs")
    config = RunnerConfig(retry_backoff=(0.0, 0.0), model_poll_seconds=0.01)
    runner = JobRunner(store, lambda: host, queue, config, normalize=normalize, find_gap=_no_gap)
    media_root = tmp_path / "media"
    media_root.mkdir(exist_ok=True)
    app = create_app(
        model_host=host,
        queue=queue,
        api_key=api_key,
        data_dir=tmp_path / "data",
        media_root=media_root if media else None,
        job_store=store,
        runner=runner,
    )
    if upload_cap is not None:
        app.state.max_upload_bytes = upload_cap
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, store=store, host=host, runner=runner, app=app, media=media_root)


def _until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.005)


def _submit_upload(svc, tmp_path, *, seconds=30, filename="talk.wav", headers=None, **fields):
    data = {**FIVE, **fields}
    files = {"file": (filename, _wav_bytes(tmp_path, seconds), "audio/wav")}
    return svc.client.post("/v1/jobs", files=files, data=data, headers=headers)


def _submit_path(svc, source_path="talk.wav", **fields):
    return svc.client.post("/v1/jobs", data={"source_path": source_path, **FIVE, **fields})


def _media(svc, tmp_path, name="talk.wav", seconds=30):
    target = svc.media / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_wav_bytes(tmp_path, seconds))
    return target


def _job(svc, job_id, **kwargs):
    return svc.client.get(f"/v1/jobs/{job_id}", **kwargs).json()


def _wait_done(svc, job_id):
    _until(lambda: _job(svc, job_id)["status"] in TERMINAL)
    return _job(svc, job_id)


def _completed_job(svc, tmp_path):
    response = _submit_upload(svc, tmp_path)
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    assert _wait_done(svc, job_id)["status"] == "completed"
    return job_id


# --- submit ---------------------------------------------------------------------


def test_upload_is_accepted_and_runs_to_completion(tmp_path):
    with _service(tmp_path) as svc:
        response = _submit_upload(svc, tmp_path, client_ref="project-7", prompt="kirkko")

        assert response.status_code == 202
        job_id = response.json()["id"]
        assert response.json() == {"id": job_id, "status": "queued"}
        assert response.headers["location"] == f"/v1/jobs/{job_id}"

        job = _wait_done(svc, job_id)
        assert job["status"] == "completed"
        assert job["phase"] == "completed"
        assert job["error"] is None
        assert job["client_ref"] == "project-7"
        assert job["progress"] == 100.0
        assert (job["current_seconds"], job["total_seconds"]) == (30.0, 30.0)
        assert (job["chunks_done"], job["chunks_total"]) == (6, 6)
        assert job["eta_seconds"] == 0.0
        assert job["params"] == {"language": "fi", "chunk_seconds": 5.0, "word_timestamps": True}
        assert job["cancel_requested"] is False
        for field in ("created_at", "started_at", "finished_at"):
            assert job[field]

        assert svc.host.calls[0]["prompt"] == "kirkko"
        assert svc.host.calls[0]["language"] == "fi"
        # The uploaded copy does not outlive the job.
        _until(lambda: not svc.store.source_dir(job_id).exists())


def test_source_path_is_resolved_under_the_media_root(tmp_path):
    with _service(tmp_path) as svc:
        _media(svc, tmp_path, "talk.wav")
        _media(svc, tmp_path, "sub/dir/other.wav", seconds=10)
        absolute = str(svc.media / "sub" / "dir" / "other.wav")

        for source_path, chunks in [("talk.wav", 6), ("sub/dir/other.wav", 2), ("sub/../talk.wav", 6), (absolute, 2)]:
            response = _submit_path(svc, source_path, language="en")
            assert response.status_code == 202, (source_path, response.text)
            job = _wait_done(svc, response.json()["id"])
            assert job["status"] == "completed", (source_path, job)
            assert job["chunks_total"] == chunks
            assert job["params"]["language"] == "en"

        assert (svc.media / "talk.wav").exists()  # the caller's file is never deleted


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, "Provide exactly one of 'file', 'source_path' or 'source_url'"),
        ({"source_path": ""}, "Provide exactly one of 'file', 'source_path' or 'source_url'"),
    ],
)
def test_exactly_one_input_is_required(tmp_path, fields, expected):
    with _service(tmp_path) as svc:
        response = svc.client.post("/v1/jobs", data=fields)
        assert response.status_code == 422
        assert response.json()["detail"] == expected

        both = svc.client.post(
            "/v1/jobs",
            files={"file": ("talk.wav", b"x", "audio/wav")},
            data={"source_path": "talk.wav"},
        )
        assert both.status_code == 422
        assert svc.store.list() == []


def test_form_parameters_are_validated(tmp_path):
    with _service(tmp_path) as svc:
        _media(svc, tmp_path)
        bad = [
            {"chunk_seconds": "4.9"},
            {"chunk_seconds": "301"},
            {"chunk_seconds": "soon"},
            {"client_ref": "x" * 201},
        ]
        for fields in bad:
            response = svc.client.post("/v1/jobs", data={"source_path": "talk.wav", **fields})
            assert response.status_code == 422, fields

        for fields in [{"chunk_seconds": "5"}, {"chunk_seconds": "300"}, {"client_ref": "x" * 200}]:
            response = svc.client.post("/v1/jobs", data={"source_path": "talk.wav", **fields})
            assert response.status_code == 202, fields

        job_id = svc.client.post(
            "/v1/jobs", data={"source_path": "talk.wav", "word_timestamps": "false", **FIVE}
        ).json()["id"]
        assert _job(svc, job_id)["params"]["word_timestamps"] is False


def test_upload_names_are_made_safe_and_reach_the_runner(tmp_path):
    seen = []

    def recording_normalize(source, wav_path, *, cancel=None):
        seen.append(Path(source))
        _copy_normalize(source, wav_path)

    with _service(tmp_path, normalize=recording_normalize) as svc:
        response = _submit_upload(svc, tmp_path, filename="..\\..\\evil name.wav")
        job_id = response.json()["id"]
        _wait_done(svc, job_id)

    assert seen == [svc.store.source_dir(job_id) / "evil_name.wav"]  # inside the job, nowhere else


@pytest.mark.parametrize(
    "given, expected",
    [
        ("sermon.mp4", "sermon.mp4"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\x\\talk 1.m4a", "talk_1.m4a"),
        ("saarna ääni.wav", "saarna_ni.wav"),
        (".hidden", "hidden"),
        ("..", "upload"),
        ("", "upload"),
        (None, "upload"),
        ("con.wav", "_con.wav"),
        ("NUL", "_NUL"),
        ("a" * 300 + ".mp3", "a" * 96 + ".mp3"),
    ],
)
def test_safe_name(given, expected):
    assert _safe_name(given) == expected


def test_empty_upload_is_rejected_and_leaves_nothing_behind(tmp_path):
    with _service(tmp_path) as svc:
        response = svc.client.post("/v1/jobs", files={"file": ("talk.wav", b"", "audio/wav")}, data=FIVE)
        assert response.status_code == 422
        assert svc.store.list() == []
        assert list(svc.store.root.iterdir()) == []


# --- source_path safety ---------------------------------------------------------


def test_source_path_needs_a_media_root(tmp_path):
    with _service(tmp_path, media=False) as svc:
        response = _submit_path(svc, "talk.wav")
        assert response.status_code == 422
        assert response.json()["detail"].startswith("source_path is not enabled")
        assert svc.store.list() == []


def test_source_path_cannot_leave_the_media_root(tmp_path):
    outside = tmp_path / "secret.wav"
    outside.write_bytes(_wav_bytes(tmp_path, 6))
    sibling = tmp_path / "media-evil"  # shares the root's name as a prefix
    sibling.mkdir()
    (sibling / "talk.wav").write_bytes(_wav_bytes(tmp_path, 6))

    with _service(tmp_path) as svc:
        _media(svc, tmp_path, "talk.wav", seconds=6)
        (svc.media / "folder").mkdir()
        attempts = [
            "../secret.wav",
            "sub/../../secret.wav",
            str(outside),
            outside.as_posix(),
            "../media-evil/talk.wav",
            str(sibling / "talk.wav"),
            "folder",  # a directory
            ".",
            "missing.wav",
            "talk.wav\x00.png",
            "\\\\server\\share\\talk.wav",
        ]
        details = set()
        for source_path in attempts:
            response = _submit_path(svc, source_path)
            assert response.status_code == 422, source_path
            details.add(response.json()["detail"])
        # One message for all of them: whether a file exists outside the root is not disclosed.
        assert details == {"source_path must name an existing file inside the media root"}
        assert svc.store.list() == []


def test_paths_outside_the_root_are_refused_without_touching_the_filesystem(tmp_path, monkeypatch):
    # Resolving a UNC path would make Windows contact that host, so an absolute path that is
    # lexically elsewhere must be turned away before any filesystem call.
    with _service(tmp_path) as svc:

        def forbidden(self, strict=False):
            raise AssertionError(f"resolve() was called for {self}")

        monkeypatch.setattr(Path, "resolve", forbidden)
        attempts = [str(tmp_path / "elsewhere.wav"), (tmp_path / "elsewhere.wav").as_posix(), "//server/share/talk.wav"]
        if os.name == "nt":
            attempts += ["\\\\server\\share\\talk.wav", "D:talk.wav"]
        for source_path in attempts:
            assert _submit_path(svc, source_path).status_code == 422, source_path


def _link_directory(link, target):
    """A symlink to a directory; on Windows without the privilege, a junction, which resolve() follows too."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if os.name != "nt":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def test_links_cannot_lead_out_of_the_media_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.wav").write_bytes(_wav_bytes(tmp_path, 6))

    with _service(tmp_path) as svc:
        try:
            _link_directory(svc.media / "link", outside)
        except (OSError, NotImplementedError, subprocess.CalledProcessError):
            pytest.skip("cannot create directory links here")
        attempts = ["link/secret.wav"]
        try:
            os.symlink(outside / "secret.wav", svc.media / "file-link.wav")
            attempts.append("file-link.wav")
        except (OSError, NotImplementedError):
            pass  # file symlinks need a privilege on Windows; the directory link already covers the escape

        for source_path in attempts:
            assert _submit_path(svc, source_path).status_code == 422, source_path
        assert svc.store.list() == []


# --- job status and listing -----------------------------------------------------


def test_unknown_and_malformed_ids_are_404(tmp_path):
    with _service(tmp_path) as svc:
        for job_id in ("0" * 32, "not-a-job", "..", "%2e%2e"):
            for method in ("get", "delete"):
                response = getattr(svc.client, method)(f"/v1/jobs/{job_id}")
                assert response.status_code == 404, (method, job_id)
            assert svc.client.get(f"/v1/jobs/{job_id}/result").status_code == 404, job_id
        assert svc.client.get(f"/v1/jobs/{'0' * 32}").json() == {"detail": "No such job"}


def test_progress_is_reported_while_a_job_runs(tmp_path):
    release = threading.Event()
    host = _Host()
    host.hook = lambda offset: offset == 10.0 and release.wait(10)  # chunk 2 hangs; chunks 0 and 1 are done

    with _service(tmp_path, host=host) as svc:
        try:
            _media(svc, tmp_path)
            job_id = _submit_path(svc).json()["id"]
            _until(lambda: 10.0 in [call["offset"] for call in host.calls])
            _until(lambda: _job(svc, job_id)["chunks_done"] == 2)

            job = _job(svc, job_id)
            assert job["status"] == "running"
            assert job["phase"] == "transcribing"
            assert (job["chunks_done"], job["chunks_total"]) == (2, 6)
            assert (job["current_seconds"], job["total_seconds"]) == (10.0, 30.0)
            assert job["progress"] == 33.3
            assert job["eta_seconds"] > 0
            assert job["finished_at"] is None
        finally:
            release.set()
        assert _wait_done(svc, job_id)["status"] == "completed"


def test_listing_can_be_filtered_by_client_ref_and_status(tmp_path):
    with _service(tmp_path) as svc:
        _media(svc, tmp_path)
        ids = {}
        for ref in ("saarna-1", "saarna-2", "saarna-1"):
            job_id = _submit_path(svc, client_ref=ref).json()["id"]
            _wait_done(svc, job_id)
            ids.setdefault(ref, []).append(job_id)

        everything = svc.client.get("/v1/jobs").json()["jobs"]
        assert [job["id"] for job in everything] == [ids["saarna-1"][1], ids["saarna-2"][0], ids["saarna-1"][0]]  # newest first
        assert {job["status"] for job in everything} == {"completed"}

        by_ref = svc.client.get("/v1/jobs", params={"client_ref": "saarna-1"}).json()["jobs"]
        assert [job["id"] for job in by_ref] == list(reversed(ids["saarna-1"]))
        assert all(job["client_ref"] == "saarna-1" for job in by_ref)
        assert svc.client.get("/v1/jobs", params={"client_ref": "nobody"}).json() == {"jobs": []}

        assert len(svc.client.get("/v1/jobs", params={"status": "completed"}).json()["jobs"]) == 3
        assert svc.client.get("/v1/jobs", params={"status": "failed"}).json() == {"jobs": []}
        both = svc.client.get("/v1/jobs", params={"status": "completed", "client_ref": "saarna-2"}).json()["jobs"]
        assert [job["id"] for job in both] == ids["saarna-2"]

        assert svc.client.get("/v1/jobs", params={"status": "bogus"}).status_code == 422


# --- results --------------------------------------------------------------------


def test_result_as_json(tmp_path):
    with _service(tmp_path) as svc:
        job_id = _completed_job(svc, tmp_path)
        response = svc.client.get(f"/v1/jobs/{job_id}/result")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        result = response.json()
        assert result["complete"] is True
        assert (result["chunks_done"], result["chunks_total"]) == (6, 6)
        assert result["language"] == "fi"
        assert result["duration_seconds"] == 30.0
        assert result["text"] == " ".join(f"chunk at {offset:g}" for offset in OFFSETS)
        assert [segment["start"] for segment in result["segments"]] == OFFSETS  # absolute time
        assert result["segments"][3]["words"][0]["start"] == 15.0
        assert svc.client.get(f"/v1/jobs/{job_id}/result", params={"format": "json"}).json() == result


def test_result_as_vtt_srt_and_text(tmp_path):
    with _service(tmp_path) as svc:
        job_id = _completed_job(svc, tmp_path)

        vtt = svc.client.get(f"/v1/jobs/{job_id}/result", params={"format": "vtt"})
        assert vtt.headers["content-type"] == "text/vtt; charset=utf-8"
        assert vtt.headers["x-job-complete"] == "true"
        assert vtt.text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nchunk at 0\n")
        assert "00:00:25.000 --> 00:00:30.000\nchunk at 25\n" in vtt.text

        srt = svc.client.get(f"/v1/jobs/{job_id}/result", params={"format": "srt"})
        assert srt.headers["content-type"] == "application/x-subrip; charset=utf-8"
        assert srt.text.startswith("1\n00:00:00,000 --> 00:00:05,000\nchunk at 0\n\n2\n")

        text = svc.client.get(f"/v1/jobs/{job_id}/result", params={"format": "text"})
        assert text.headers["content-type"] == "text/plain; charset=utf-8"
        assert text.text == " ".join(f"chunk at {offset:g}" for offset in OFFSETS)

        # Cue shaping options reach make_cues: a line limit of 5 wraps "chunk at 0" onto two lines.
        narrow = svc.client.get(f"/v1/jobs/{job_id}/result", params={"format": "srt", "max_line_chars": 5})
        assert "chunk\nat 0\n" in narrow.text


def test_result_as_youtube_needs_a_start_time(tmp_path):
    with _service(tmp_path) as svc:
        job_id = _completed_job(svc, tmp_path)
        url = f"/v1/jobs/{job_id}/result"

        missing = svc.client.get(url, params={"format": "youtube"})
        assert missing.status_code == 422
        assert "start_time" in missing.json()["detail"]
        assert svc.client.get(url, params={"format": "youtube", "start_time": "yesterday"}).status_code == 422

        response = svc.client.get(url, params={"format": "youtube", "start_time": "2026-01-01T12:00:00Z"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.text.startswith("2026-01-01T12:00:00.000\nchunk at 0\n2026-01-01T12:00:05.000\nchunk at 5\n")
        assert response.text.endswith("2026-01-01T12:00:25.000\nchunk at 25\n")

        tagged = svc.client.get(
            url,
            params={"format": "youtube", "start_time": "2026-01-01T12:00:00Z", "region": "reg2", "cue": "cue9"},
        )
        assert tagged.text.startswith("2026-01-01T12:00:00.000 region:reg2#cue9\nchunk at 0\n")

        bad_region = svc.client.get(
            url, params={"format": "youtube", "start_time": "2026-01-01T12:00:00Z", "region": "two words"}
        )
        assert bad_region.status_code == 422


def test_youtube_start_time_offset_survives_an_unencoded_plus(tmp_path):
    with _service(tmp_path) as svc:
        job_id = _completed_job(svc, tmp_path)
        # curl users write "+02:00" without encoding it; the server sees a space.
        response = svc.client.get(f"/v1/jobs/{job_id}/result?format=youtube&start_time=2026-01-01T14:00:00+02:00")

        assert response.status_code == 200
        assert response.text.startswith("2026-01-01T12:00:00.000\n")  # 14:00 at +02:00 is 12:00 UTC


def test_unknown_format_and_bad_cue_options_are_422(tmp_path):
    with _service(tmp_path) as svc:
        job_id = _completed_job(svc, tmp_path)
        url = f"/v1/jobs/{job_id}/result"

        response = svc.client.get(url, params={"format": "docx"})
        assert response.status_code == 422
        assert "json" in response.json()["detail"]
        assert svc.client.get(url, params={"format": "vtt", "max_cue_duration": 0}).status_code == 422
        assert svc.client.get(url, params={"format": "vtt", "max_line_chars": 0}).status_code == 422


def test_result_of_an_unfinished_job_is_409_unless_partial_is_asked_for(tmp_path):
    release = threading.Event()
    host = _Host()
    host.hook = lambda offset: offset == 10.0 and release.wait(10)

    with _service(tmp_path, host=host) as svc:
        try:
            _media(svc, tmp_path)
            job_id = _submit_path(svc).json()["id"]
            _until(lambda: _job(svc, job_id)["chunks_done"] == 2)
            url = f"/v1/jobs/{job_id}/result"

            for params in ({}, {"format": "vtt"}, {"partial": "false"}):
                response = svc.client.get(url, params=params)
                assert response.status_code == 409
                assert response.json() == {"detail": "Job is not finished", "status": "running"}

            partial = svc.client.get(url, params={"partial": "true"})
            assert partial.status_code == 200
            body = partial.json()
            assert body["complete"] is False
            assert (body["chunks_done"], body["chunks_total"]) == (2, 6)
            assert body["text"] == "chunk at 0 chunk at 5"

            for fmt in ("vtt", "srt", "text"):
                response = svc.client.get(url, params={"format": fmt, "partial": "1"})
                assert response.status_code == 200
                assert response.headers["x-job-complete"] == "false"
            youtube = svc.client.get(
                url, params={"format": "youtube", "partial": "true", "start_time": "2026-01-01T12:00:00Z"}
            )
            assert youtube.headers["x-job-complete"] == "false"
        finally:
            release.set()

        assert _wait_done(svc, job_id)["status"] == "completed"
        assert svc.client.get(url, params={"partial": "true"}).json()["complete"] is True
        assert svc.client.get(url).status_code == 200


def test_result_of_a_failed_job_is_readable_only_as_partial(tmp_path):
    host = _Host()

    def explode(offset):
        if offset == 15.0:
            raise RuntimeError("model exploded")

    host.hook = explode

    with _service(tmp_path, host=host) as svc:
        _media(svc, tmp_path)
        job_id = _submit_path(svc).json()["id"]
        job = _wait_done(svc, job_id)

        assert job["status"] == "failed"
        assert job["error"] == "Chunk 3 failed after 3 attempts: RuntimeError: model exploded"
        assert job["eta_seconds"] is None
        assert (job["chunks_done"], job["chunks_total"]) == (3, 6)

        url = f"/v1/jobs/{job_id}/result"
        response = svc.client.get(url)
        assert response.status_code == 409
        assert response.json() == {"detail": "Job is not finished", "status": "failed"}

        partial = svc.client.get(url, params={"partial": "true"}).json()
        assert partial["complete"] is False
        assert partial["text"] == "chunk at 0 chunk at 5 chunk at 10"


# --- delete ---------------------------------------------------------------------


def test_delete_removes_a_finished_job(tmp_path):
    with _service(tmp_path) as svc:
        job_id = _completed_job(svc, tmp_path)

        assert svc.client.delete(f"/v1/jobs/{job_id}").status_code == 204
        assert svc.client.get(f"/v1/jobs/{job_id}").status_code == 404
        assert svc.client.get(f"/v1/jobs/{job_id}/result").status_code == 404
        assert not svc.store.job_dir(job_id).exists()
        assert svc.client.delete(f"/v1/jobs/{job_id}").status_code == 404


def test_delete_a_queued_job_removes_it_before_it_ever_runs(tmp_path):
    release = threading.Event()
    host = _Host()
    host.hook = lambda offset: release.wait(10)

    with _service(tmp_path, host=host) as svc:
        try:
            _media(svc, tmp_path)
            first = _submit_path(svc).json()["id"]
            _until(lambda: host.calls)  # the first job holds the worker
            second = _submit_path(svc).json()["id"]
            assert _job(svc, second)["status"] == "queued"

            assert svc.client.delete(f"/v1/jobs/{second}").status_code == 204
            assert svc.client.get(f"/v1/jobs/{second}").status_code == 404
        finally:
            release.set()

        assert _wait_done(svc, first)["status"] == "completed"
        assert [call["offset"] for call in host.calls] == OFFSETS  # only the first job ran


def test_delete_a_running_job_cancels_it_and_the_runner_removes_everything(tmp_path):
    release = threading.Event()
    host = _Host()
    host.hook = lambda offset: offset == 5.0 and release.wait(10)

    with _service(tmp_path, host=host) as svc:
        try:
            job_id = _submit_upload(svc, tmp_path).json()["id"]
            _until(lambda: 5.0 in [call["offset"] for call in host.calls])

            response = svc.client.delete(f"/v1/jobs/{job_id}")
            assert response.status_code == 202
            assert response.json() == {"id": job_id, "status": "running", "cancel_requested": True}
            assert _job(svc, job_id)["cancel_requested"] is True
        finally:
            release.set()

        _until(lambda: svc.client.get(f"/v1/jobs/{job_id}").status_code == 404)
        _until(lambda: not svc.store.job_dir(job_id).exists())
        assert [call["offset"] for call in host.calls] == [0.0, 5.0]  # it stopped at the chunk boundary
        assert svc.store.list() == []


# --- upload limits --------------------------------------------------------------


def test_upload_larger_than_the_cap_is_rejected_and_leaves_nothing_behind(tmp_path):
    payload = _wav_bytes(tmp_path, 6)
    with _service(tmp_path, upload_cap=len(payload) - 1) as svc:
        response = svc.client.post("/v1/jobs", files={"file": ("talk.wav", payload, "audio/wav")}, data=FIVE)

        assert response.status_code == 413  # streamed past the cap: the framing allowance let it in
        assert "too large" in response.json()["detail"]
        assert svc.store.list() == []
        assert list(svc.store.root.iterdir()) == []  # no half-written job directory


def test_upload_exactly_at_the_cap_is_accepted(tmp_path):
    payload = _wav_bytes(tmp_path, 6)
    with _service(tmp_path, upload_cap=len(payload)) as svc:
        response = svc.client.post("/v1/jobs", files={"file": ("talk.wav", payload, "audio/wav")}, data=FIVE)
        assert response.status_code == 202


def test_a_huge_announced_upload_is_rejected_before_the_body_is_read(tmp_path):
    with _service(tmp_path, upload_cap=1000) as svc:
        sent = asyncio.run(_raw_post(svc.app, {"content-length": str(50 * 1024 * 1024 * 1024)}))

    assert sent["status"] == 413
    assert sent["body_reads"] == 0  # the announced size alone was enough
    assert svc.store.list() == []


def _multipart_file_head(boundary="b"):
    crlf = bytes([13, 10])
    line = f'--{boundary}{crlf.decode()}Content-Disposition: form-data; name="file"; filename="talk.wav"'
    return line.encode() + crlf + crlf


def test_a_body_without_content_length_is_cut_off_once_it_exceeds_the_cap(tmp_path):
    body = _multipart_file_head("xxxxboundaryxxxx") + bytes(3 * 1024 * 1024)

    with _service(tmp_path, upload_cap=1000) as svc:
        response = svc.client.post(
            "/v1/jobs",
            content=(chunk for chunk in [body[:100], body[100:]]),  # chunked transfer: no Content-Length
            headers={"content-type": "multipart/form-data; boundary=xxxxboundaryxxxx"},
        )

        assert response.status_code == 413
        assert svc.store.list() == []
        assert svc.client.get("/health").status_code == 200  # the app is fine afterwards


def test_a_streamed_body_over_the_cap_is_answered_413_exactly_once(tmp_path):
    chunks = [_multipart_file_head()] + [bytes(100_000)] * 30  # 3 MB; the cap plus allowance is about 1 MB

    with _service(tmp_path, upload_cap=1000) as svc:
        sent = asyncio.run(_raw_post(svc.app, {"content-type": "multipart/form-data; boundary=b"}, body_chunks=chunks))

    assert sent["status"] == 413
    assert sent["responses"] == 1  # whatever the app tried to say after the cut-off was dropped
    assert sent["body_reads"] < len(chunks)  # cut off part way, not read to the end
    assert svc.store.list() == []


def test_upload_cap_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDITOR_STT_MAX_UPLOAD_MB", "0.5")
    app = create_app(model_host=_Host(), queue=InferenceQueue(), data_dir=tmp_path / "data")
    assert app.state.max_upload_bytes == 512 * 1024

    monkeypatch.delenv("AUDITOR_STT_MAX_UPLOAD_MB")
    app = create_app(model_host=_Host(), queue=InferenceQueue(), data_dir=tmp_path / "data")
    assert app.state.max_upload_bytes == 2048 * 1024 * 1024


# --- authentication -------------------------------------------------------------


def test_every_job_route_requires_the_key_when_one_is_configured(tmp_path):
    with _service(tmp_path, api_key=KEY) as svc:
        _media(svc, tmp_path)
        job_id = "0" * 32
        rejected = [
            svc.client.get("/v1/jobs"),
            svc.client.get(f"/v1/jobs/{job_id}"),
            svc.client.get(f"/v1/jobs/{job_id}/result"),
            svc.client.delete(f"/v1/jobs/{job_id}"),
            svc.client.post("/v1/jobs", data={"source_path": "talk.wav"}),
            svc.client.post("/v1/jobs", data={"source_path": "talk.wav"}, headers={"Authorization": "Bearer nope"}),
            svc.client.post("/v1/jobs", files={"file": ("talk.wav", b"x", "audio/wav")}),
        ]
        for response in rejected:
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
        assert svc.store.list() == []

        headers = {"Authorization": f"Bearer {KEY}"}
        assert svc.client.get("/v1/jobs", headers=headers).status_code == 200
        response = svc.client.post("/v1/jobs", data={"source_path": "talk.wav", **FIVE}, headers=headers)
        assert response.status_code == 202
        assert svc.client.get("/health").status_code == 200  # stays open for probes


def test_an_unauthenticated_upload_is_refused_before_its_body_is_read(tmp_path):
    with _service(tmp_path, api_key=KEY) as svc:
        for headers in ({}, {"authorization": "Bearer wrong"}, {"authorization": f"Basic {KEY}"}):
            sent = asyncio.run(_raw_post(svc.app, {"content-length": "10485760", **headers}))
            assert sent["status"] == 401, headers
            assert sent["headers"]["www-authenticate"] == "Bearer"
            assert sent["body_reads"] == 0
        assert svc.store.list() == []


# --- jobs not configured --------------------------------------------------------


def test_without_a_data_dir_nothing_is_created_and_job_routes_answer_503(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(model_host=_Host(), queue=InferenceQueue())
    assert app.state.job_store is None and app.state.job_runner is None

    with TestClient(app) as client:
        responses = [
            client.get("/v1/jobs"),
            client.get(f"/v1/jobs/{'0' * 32}"),
            client.get(f"/v1/jobs/{'0' * 32}/result"),
            client.delete(f"/v1/jobs/{'0' * 32}"),
            client.post("/v1/jobs", data={"source_path": "talk.wav"}),
            client.post("/v1/jobs", files={"file": ("talk.wav", b"x", "audio/wav")}),
        ]
        for response in responses:
            assert response.status_code == 503
            assert response.json() == {"detail": NOT_CONFIGURED}
        assert client.get("/health").status_code == 200
        assert client.post("/inference", files={"file": ("clip.wav", b"x", "audio/wav")}).status_code == 200

    assert list(tmp_path.iterdir()) == []  # no data directory, nothing else either


def test_the_key_is_checked_before_the_not_configured_answer(tmp_path):
    app = create_app(model_host=_Host(), queue=InferenceQueue(), api_key=KEY)
    with TestClient(app) as client:
        assert client.get("/v1/jobs").status_code == 401
        assert client.post("/v1/jobs", data={"source_path": "x"}).status_code == 401
        assert client.get("/v1/jobs", headers={"Authorization": f"Bearer {KEY}"}).status_code == 503


def test_data_dir_and_media_root_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDITOR_STT_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("AUDITOR_STT_MEDIA_ROOT", str(tmp_path / "volume"))
    (tmp_path / "volume").mkdir()

    app = create_app(model_host=_Host(), queue=InferenceQueue())

    assert app.state.job_store is not None and app.state.job_runner is not None
    assert app.state.job_store.root == tmp_path / "state" / "jobs"
    assert app.state.media_root == (tmp_path / "volume").resolve()
    assert (tmp_path / "state" / "jobs").is_dir()


def test_the_runner_is_started_and_stopped_with_the_app(tmp_path):
    with _service(tmp_path) as svc:
        assert svc.runner._worker is not None and not svc.runner._worker.done()
    assert svc.runner._worker is None  # stopped by the lifespan


# --- the whole stack with real ffmpeg and real chunk planning -------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_end_to_end_with_real_ffmpeg_and_vad_chunk_planning(tmp_path, monkeypatch):
    source = tmp_path / "volume" / "sermon.m4a"
    source.parent.mkdir()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=14"]
        + ["-ar", "44100", "-ac", "2", str(source)],
        check=True,
    )
    monkeypatch.setenv("AUDITOR_STT_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("AUDITOR_STT_MEDIA_ROOT", str(tmp_path / "volume"))
    host = _Host()

    # No injected runner: the defaults are real ffmpeg normalisation and Silero-based chunk planning.
    with TestClient(create_app(model_host=host, queue=InferenceQueue())) as client:
        response = client.post("/v1/jobs", data={"source_path": "sermon.m4a", "chunk_seconds": "5"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        _until(lambda: client.get(f"/v1/jobs/{job_id}").json()["status"] in TERMINAL, timeout=60)

        job = client.get(f"/v1/jobs/{job_id}").json()
        assert job["status"] == "completed", job
        result = client.get(f"/v1/jobs/{job_id}/result").json()

    assert result["complete"] is True
    assert result["duration_seconds"] == pytest.approx(14.0, abs=0.2)
    starts = [segment["start"] for segment in result["segments"]]
    assert starts == sorted(starts) and starts[0] == 0.0 and len(starts) >= 2
    assert all(0 <= start < 14 for start in starts)
    assert source.exists()
    assert not (tmp_path / "state" / "jobs" / job_id / "audio.wav").exists()


# --- raw ASGI helper ------------------------------------------------------------


async def _raw_post(app, headers, body_chunks=()):
    """POST /v1/jobs straight into the ASGI app, counting how often its body is read."""
    reads = 0
    pending = list(body_chunks)

    async def receive():
        nonlocal reads
        reads += 1
        if not pending:
            return {"type": "http.disconnect"} if body_chunks else {"type": "http.request", "body": b""}
        return {"type": "http.request", "body": pending.pop(0), "more_body": bool(pending)}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/jobs",
        "raw_path": b"/v1/jobs",
        "root_path": "",
        "query_string": b"",
        "headers": [(name.encode(), value.encode()) for name, value in headers.items()],
        "client": ("testclient", 5000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)

    starts = [message for message in sent if message["type"] == "http.response.start"]
    start = starts[0]
    return {
        "responses": len(starts),
        "status": start["status"],
        "headers": {name.decode(): value.decode() for name, value in start["headers"]},
        "body_reads": reads,
    }


# --- source_url ------------------------------------------------------------


@contextmanager
def _file_server(routes):
    """Serves {path: (status, body, headers)} on 127.0.0.1 and yields the base URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, body, headers = routes.get(self.path.split("?")[0], (404, b"", {}))
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _submit_url(svc, url, **fields):
    return svc.client.post("/v1/jobs", data={"source_url": url, **FIVE, **fields})


def test_source_url_is_fetched_from_an_allowed_host(tmp_path):
    body = _wav_bytes(tmp_path, 30)
    with _file_server({"/bucket/talk.wav": (200, body, {})}) as base, _service(tmp_path) as svc:
        svc.app.state.source_url_hosts = ["127.0.0.1"]
        response = _submit_url(svc, f"{base}/bucket/talk.wav?X-Amz-Signature=abc", language="en")
        assert response.status_code == 202, response.text
        job = _wait_done(svc, response.json()["id"])
        assert job["status"] == "completed"
        assert job["chunks_total"] == 6
        # Like an upload, the fetched copy is removed when the job ends.
        assert not list(svc.store.source_dir(response.json()["id"]).glob("*"))


def test_source_url_is_disabled_without_host_list(tmp_path):
    with _service(tmp_path) as svc:
        response = _submit_url(svc, "http://127.0.0.1:9/talk.wav")
        assert response.status_code == 422
        assert "AUDITOR_STT_SOURCE_URL_HOSTS" in response.json()["detail"]


@pytest.mark.parametrize(
    "url",
    [
        "http://other.example/talk.wav",  # host not listed
        "ftp://127.0.0.1/talk.wav",
        "file:///etc/passwd",
        "http://user:pw@127.0.0.1/talk.wav",
        "http://127.0.0.1:notaport/talk.wav",
        "not a url",
    ],
)
def test_source_url_refuses_unlisted_or_odd_urls(tmp_path, url):
    with _service(tmp_path) as svc:
        svc.app.state.source_url_hosts = ["127.0.0.1"]
        response = _submit_url(svc, url)
        assert response.status_code == 422, url
        assert not list((tmp_path / "data" / "jobs").glob("*"))


def test_source_url_failures_leave_no_job(tmp_path):
    routes = {
        "/redirect": (302, b"", {"Location": "http://169.254.169.254/latest/meta-data"}),
        "/empty": (200, b"", {}),
        "/big": (200, b"x" * 2048, {}),
    }
    with _file_server(routes) as base, _service(tmp_path, upload_cap=1024) as svc:
        svc.app.state.source_url_hosts = ["127.0.0.1"]
        assert _submit_url(svc, f"{base}/missing").status_code == 422  # 404 from the server
        assert _submit_url(svc, f"{base}/redirect").status_code == 422  # redirects are not followed
        assert _submit_url(svc, f"{base}/empty").status_code == 422
        assert _submit_url(svc, f"{base}/big").status_code == 413
        assert _submit_url(svc, "http://127.0.0.1:1/unreachable.wav").status_code == 422
        assert not list((tmp_path / "data" / "jobs").glob("*"))
