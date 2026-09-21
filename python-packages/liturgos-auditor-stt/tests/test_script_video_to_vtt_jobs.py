"""Tests for scripts/video-to-vtt-jobs.py"""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

# Load the script as a module
script_path = Path(__file__).resolve().parents[3] / "scripts" / "video-to-vtt-jobs.py"
spec = importlib.util.spec_from_file_location("video_to_vtt", script_path)
video_to_vtt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_to_vtt)


class FakeJobServer(http.server.BaseHTTPRequestHandler):
    """Fake service that implements the jobs API."""

    # Class-level state shared across requests
    jobs = {}  # job_id -> {"status": ..., "progress": ..., "chunks": ...}
    requests_log = []  # Record of all requests
    api_key_required = None
    poll_count = 0

    def do_POST(self):
        """Handle POST /v1/jobs (job submission)."""
        if self.path != "/v1/jobs":
            self.send_error(404)
            return

        # Check API key if required
        if self.api_key_required:
            auth_header = self.headers.get("Authorization", "")
            if auth_header != f"Bearer {self.api_key_required}":
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Unauthorized"}).encode())
                return

        # Parse multipart
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Record the request
        FakeJobServer.requests_log.append(
            {
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "body_length": len(body),
                "has_file_part": b'name="file"' in body,
                "has_source_path": b'name="source_path"' in body,
            }
        )

        # Create job ID (32 hex characters)
        job_id = f"test{len(FakeJobServer.jobs):028x}"
        FakeJobServer.jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "chunks_done": 0,
            "chunks_total": 10,
            "eta_seconds": None,
            "error": None,
        }

        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"id": job_id, "status": "queued"}).encode())

    def do_GET(self):
        """Handle GET /v1/jobs/{id} (status) or /v1/jobs/{id}/result (fetch)."""
        # Check API key if required
        if self.api_key_required:
            auth_header = self.headers.get("Authorization", "")
            if auth_header != f"Bearer {self.api_key_required}":
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Unauthorized"}).encode())
                return

        if self.path.split("?")[0] == "/v1/jobs":
            self.send_job_list()
            return

        if self.path.startswith("/v1/jobs/"):
            # Strip query string from path for path-based routing
            path_only = self.path.split("?")[0]
            parts = path_only.split("/")
            job_id = parts[3] if len(parts) > 3 else None
            is_result = len(parts) > 4 and parts[4] == "result"

            if not job_id or job_id not in FakeJobServer.jobs:
                self.send_error(404)
                return

            if is_result:
                # Fetch result
                job = FakeJobServer.jobs[job_id]
                FakeJobServer.requests_log.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
                fmt = "vtt"
                if "format=json" in self.path:
                    fmt = "json"
                elif "format=srt" in self.path:
                    fmt = "srt"
                elif "format=youtube" in self.path:
                    fmt = "youtube"

                if fmt == "json":
                    content = json.dumps({"segments": [{"text": "test", "start": 0, "end": 1}]})
                elif fmt == "youtube":
                    content = "0:00:00,000 --> 0:00:01,000\ntest"
                elif fmt == "srt":
                    content = "1\n00:00:00,000 --> 00:00:01,000\ntest\n"
                else:  # vtt
                    content = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\ntest\n"

                self.send_response(200)
                self.send_header("Content-Type", "text/plain" if fmt != "json" else "application/json")
                self.send_header("X-Job-Complete", "true" if job["status"] == "completed" else "false")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                # Get status
                job = FakeJobServer.jobs[job_id]

                # Simulate progress
                FakeJobServer.poll_count += 1
                if FakeJobServer.poll_count == 1:
                    job["status"] = "running"
                    job["progress"] = 30
                    job["chunks_done"] = 3
                elif FakeJobServer.poll_count == 2:
                    job["status"] = "running"
                    job["progress"] = 70
                    job["chunks_done"] = 7
                elif FakeJobServer.poll_count >= 3:
                    job["status"] = "completed"
                    job["progress"] = 100
                    job["chunks_done"] = 10

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(job).encode())
        else:
            self.send_error(404)

    def send_job_list(self):
        """GET /v1/jobs: what the client's pre-flight check calls."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"jobs": []}).encode())

    def do_DELETE(self):
        """Handle DELETE /v1/jobs/{id}."""
        if self.api_key_required:
            auth_header = self.headers.get("Authorization", "")
            if auth_header != f"Bearer {self.api_key_required}":
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Unauthorized"}).encode())
                return

        parts = self.path.split("/")
        job_id = parts[3] if len(parts) > 3 else None

        if not job_id or job_id not in FakeJobServer.jobs:
            self.send_error(404)
            return

        del FakeJobServer.jobs[job_id]
        self.send_response(204)
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress logging."""
        pass


class FakeJobServerThread:
    """Thread-based server for testing."""

    def __init__(self, port=0, api_key=None, handler=None):
        self.port = port
        self.api_key = api_key
        self.handler = handler or FakeJobServer
        self.server = None
        self.thread = None
        self.url = None

    def start(self):
        """Start the server."""
        FakeJobServer.api_key_required = self.api_key
        FakeJobServer.jobs = {}
        FakeJobServer.requests_log = []
        FakeJobServer.poll_count = 0

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), self.handler)
        self.port = self.server.server_port
        self.url = f"http://127.0.0.1:{self.port}"

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def get_job_ids(self):
        """Get all job IDs."""
        return list(FakeJobServer.jobs.keys())

    def requests(self):
        """Get recorded requests."""
        return FakeJobServer.requests_log


@pytest.fixture
def fake_server():
    """Create a fake server for testing."""
    server = FakeJobServerThread()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def fake_server_with_auth():
    """Create a fake server with authentication."""
    server = FakeJobServerThread(api_key="test-secret-key")
    server.start()
    yield server
    server.stop()


def test_successful_upload_and_transcribe(tmp_path, fake_server, monkeypatch):
    """Test successful job submission, polling, and result fetch with file upload."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.vtt"

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
        "--no-extract",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0
    assert output_file.exists()
    assert "WEBVTT" in output_file.read_text()
    assert not output_file.with_suffix(".vtt.tmp").exists()

    # Verify job was deleted
    assert len(fake_server.get_job_ids()) == 0

    # Verify file was uploaded (not source_path)
    requests = fake_server.requests()
    assert any(req["has_file_part"] for req in requests)


def test_keep_job_flag(tmp_path, fake_server, monkeypatch):
    """Test that --keep-job prevents job deletion."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.vtt"

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
        "--keep-job",
        "--no-extract",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0
    assert output_file.exists()

    # Verify job was NOT deleted
    assert len(fake_server.get_job_ids()) == 1


def test_source_path_field(tmp_path, fake_server, monkeypatch):
    """Test --source-path sends source_path field without file upload."""
    output_file = tmp_path / "output.vtt"

    argv = [
        "video-to-vtt.py",
        "--source-path",
        "/server/media/file.wav",
        str(output_file),
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0

    # Verify source_path was sent
    posts = [req for req in fake_server.requests() if req["method"] == "POST"]
    assert any(req["has_source_path"] for req in posts)
    assert not any(req["has_file_part"] for req in posts)


def test_resume_job(tmp_path, fake_server, monkeypatch):
    """Test --resume skips submission and just polls."""
    # Create a dummy video file (not used when resuming, but needed for positional arg)
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy")
    output_file = tmp_path / "output.vtt"

    # Manually create a job in the fake server's jobs dict
    job_id = f"test{len(FakeJobServer.jobs):028x}"
    FakeJobServer.jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "chunks_done": 0,
        "chunks_total": 10,
        "eta_seconds": None,
        "error": None,
    }

    # Now resume it
    initial_requests_count = len(fake_server.requests())
    argv = [
        "video-to-vtt.py",
        str(video_file),  # Pass as first positional (VIDEO)
        str(output_file),  # Pass as second positional (OUTPUT)
        "--resume",
        job_id,
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0
    assert output_file.exists()

    # Verify no submission request was made
    new_requests = fake_server.requests()[initial_requests_count:]
    assert all(req["method"] != "POST" for req in new_requests)


def test_source_path_with_lone_positional_writes_that_output(tmp_path, fake_server, monkeypatch):
    """`--source-path P OUT`: OUT is the output, not a VIDEO to ignore (regression: it was written to ./P-stem.vtt)."""
    monkeypatch.chdir(tmp_path)  # where the old, derived output name would have landed
    output_file = tmp_path / "chosen" / "output.vtt"
    output_file.parent.mkdir()

    argv = [
        "video-to-vtt.py",
        "--source-path",
        "sermon.mp4",
        str(output_file),
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert video_to_vtt.main() == 0
    assert output_file.exists()
    assert not (tmp_path / "sermon.vtt").exists()


def test_resume_with_lone_positional_is_the_output(tmp_path, fake_server, monkeypatch):
    """The documented `--resume JOB_ID OUTPUT` (regression: "OUTPUT is required with --resume")."""
    output_file = tmp_path / "output.vtt"
    job_id = f"test{len(FakeJobServer.jobs):028x}"
    FakeJobServer.jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "chunks_done": 0,
        "chunks_total": 10,
        "eta_seconds": None,
        "error": None,
    }

    argv = ["video-to-vtt.py", "--resume", job_id, str(output_file), "--url", fake_server.url, "--poll-interval", "0.01"]
    monkeypatch.setattr(sys, "argv", argv)

    assert video_to_vtt.main() == 0
    assert output_file.exists()
    assert all(req["method"] != "POST" for req in fake_server.requests())


def test_output_is_written_with_lf_line_endings(tmp_path):
    """The file holds exactly what the service sent: no "\\n" -> "\\r\\n" translation on Windows."""
    target = tmp_path / "out.txt"
    video_to_vtt.write_atomic(target, "2026-01-01T10:00:00.000\nhello\n")
    assert target.read_bytes() == b"2026-01-01T10:00:00.000\nhello\n"
    assert not target.with_suffix(".txt.tmp").exists()


def test_api_key_header_when_set(tmp_path, fake_server_with_auth, monkeypatch):
    """Test Authorization header is sent when API key env var is set."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.vtt"

    monkeypatch.setenv("AUDITOR_STT_API_KEY", "test-secret-key")

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--url",
        fake_server_with_auth.url,
        "--poll-interval",
        "0.01",
        "--no-extract",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0

    # Verify Authorization header was sent
    requests = fake_server_with_auth.requests()
    assert any("Authorization" in req["headers"] for req in requests)


def test_api_key_header_not_sent_when_unset(tmp_path, fake_server, monkeypatch):
    """Test Authorization header is not sent when API key env var is unset."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.vtt"

    # Ensure env var is not set
    monkeypatch.delenv("AUDITOR_STT_API_KEY", raising=False)

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
        "--no-extract",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0

    # Verify Authorization header was NOT sent
    requests = fake_server.requests()
    assert not any("Authorization" in req["headers"] for req in requests)


def test_youtube_format_requires_start_time(tmp_path, fake_server, monkeypatch):
    """Test that youtube format without --start-time is a usage error."""
    video_file = tmp_path / "test.wav"
    video_file.write_text("dummy video")
    output_file = tmp_path / "output.txt"

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--format",
        "youtube",
        "--url",
        fake_server.url,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Should fail due to missing --start-time (argparse.error exits with 2)
    with pytest.raises(SystemExit) as exc_info:
        video_to_vtt.main()
    assert exc_info.value.code == 2


def test_youtube_format_with_start_time(tmp_path, fake_server, monkeypatch):
    """Test youtube format with --start-time works."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.txt"

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--format",
        "youtube",
        "--start-time",
        "2024-01-01T00:00:00Z",
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
        "--no-extract",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0
    assert output_file.exists()


def test_failed_job_writes_partial(tmp_path, monkeypatch):
    """Test that failed job saves partial result with .partial suffix."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.vtt"

    # Modify server to return failed status
    def get_handler(*args, **kwargs):
        class FailedJobHandler(FakeJobServer):
            def do_GET(self):
                if self.api_key_required:
                    auth_header = self.headers.get("Authorization", "")
                    if auth_header != f"Bearer {self.api_key_required}":
                        self.send_response(401)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"detail": "Unauthorized"}).encode())
                        return

                if self.path.split("?")[0] == "/v1/jobs":
                    self.send_job_list()
                    return

                if self.path.startswith("/v1/jobs/"):
                    # Strip query string from path for path-based routing
                    path_only = self.path.split("?")[0]
                    parts = path_only.split("/")
                    job_id = parts[3] if len(parts) > 3 else None
                    is_result = len(parts) > 4 and parts[4] == "result"

                    if not job_id or job_id not in FakeJobServer.jobs:
                        self.send_error(404)
                        return

                    if is_result:
                        content = "partial result"
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.send_header("X-Job-Complete", "false")
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content.encode())
                    else:
                        job = FakeJobServer.jobs[job_id]
                        if job["status"] == "queued":
                            job["status"] = "failed"
                            job["error"] = "Test error"

                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(job).encode())
                else:
                    self.send_error(404)

        return FailedJobHandler(*args, **kwargs)

    server = FakeJobServerThread()
    server.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), get_handler)
    server.port = server.server.server_port
    server.url = f"http://127.0.0.1:{server.port}"
    server.thread = threading.Thread(target=server.server.serve_forever, daemon=True)
    server.thread.start()

    try:
        argv = [
            "video-to-vtt.py",
            str(video_file),
            str(output_file),
            "--url",
            server.url,
            "--poll-interval",
            "0.01",
            "--no-extract",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        result = video_to_vtt.main()

        assert result == 3  # Exit code 3 for failed job
        partial_file = output_file.with_suffix(".vtt.partial")
        assert partial_file.exists()
        assert "partial result" in partial_file.read_text()
    finally:
        server.stop()


def test_output_written_atomically(tmp_path, fake_server, monkeypatch):
    """Test that output is written atomically (no .tmp left)."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.vtt"

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
        "--no-extract",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0
    assert output_file.exists()
    assert not output_file.with_suffix(".vtt.tmp").exists()


def test_json_out_flag(tmp_path, fake_server, monkeypatch):
    """Test --json-out also saves JSON transcript."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.vtt"
    json_file = tmp_path / "output.json"

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--json-out",
        str(json_file),
        "--url",
        fake_server.url,
        "--poll-interval",
        "0.01",
        "--no-extract",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0
    assert output_file.exists()
    assert json_file.exists()
    assert "segments" in json_file.read_text()


def test_transient_error_retry(tmp_path, monkeypatch):
    """Test that transient errors are retried."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy video data")
    output_file = tmp_path / "output.vtt"

    call_count = [0]

    def failing_submit(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 2:
            raise ConnectionRefusedError("test error")
        # Return a valid response on retry
        return "test000000000000000000000000"

    # Mock extract_audio to avoid trying to run ffmpeg
    monkeypatch.setattr(video_to_vtt, "extract_audio", lambda *args, **kwargs: None)
    monkeypatch.setattr(video_to_vtt, "submit_job", failing_submit)
    monkeypatch.setattr(video_to_vtt, "check_service", lambda *args, **kwargs: None)
    monkeypatch.setattr(video_to_vtt.time, "sleep", lambda seconds: None)

    # Mock other functions to avoid network calls
    monkeypatch.setattr(
        video_to_vtt,
        "get_job_status",
        lambda *args, **kwargs: {
            "status": "completed",
            "progress": 100,
            "chunks_done": 10,
            "chunks_total": 10,
            "eta_seconds": None,
            "error": None,
        },
    )
    monkeypatch.setattr(
        video_to_vtt,
        "fetch_job_result",
        lambda *args, **kwargs: ("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\ntest", True),
    )
    monkeypatch.setattr(video_to_vtt, "delete_job", lambda *args, **kwargs: None)

    argv = [
        "video-to-vtt.py",
        str(video_file),
        str(output_file),
        "--url",
        "http://localhost:8090",
        "--poll-interval",
        "0.01",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    result = video_to_vtt.main()

    assert result == 0
    assert call_count[0] >= 2  # Should have retried


def test_rejected_api_key_fails_before_extraction_or_upload(tmp_path, fake_server_with_auth, monkeypatch, capsys):
    """A wrong key is reported by the pre-flight check, before any ffmpeg work or upload."""
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy")
    extracted = []
    monkeypatch.setattr(video_to_vtt, "extract_audio", lambda *args, **kwargs: extracted.append(1))
    monkeypatch.delenv("AUDITOR_STT_API_KEY", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["video-to-vtt.py", str(video_file), str(tmp_path / "out.vtt"), "--url", fake_server_with_auth.url]
    )

    assert video_to_vtt.main() == 1

    assert "API key" in capsys.readouterr().err
    assert extracted == []
    assert [r for r in fake_server_with_auth.requests() if r["method"] == "POST"] == []


class _Flaky503(FakeJobServer):
    """Answers the first status polls with 503, like a proxy restarting in front of the service."""

    failures_left = 2

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/v1/jobs/") and not path.endswith("/result") and _Flaky503.failures_left > 0:
            _Flaky503.failures_left -= 1
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "restarting"}).encode())
            return
        super().do_GET()


def test_transient_503_while_polling_is_retried(tmp_path, monkeypatch):
    """The job keeps running server-side, so a brief 503 must not end the client."""
    _Flaky503.failures_left = 2
    server = FakeJobServerThread(handler=_Flaky503)
    server.start()
    try:
        video_file = tmp_path / "test.wav"
        video_file.write_bytes(b"dummy")
        output_file = tmp_path / "out.vtt"
        monkeypatch.setattr(video_to_vtt.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(
            sys,
            "argv",
            ["video-to-vtt.py", str(video_file), str(output_file), "--url", server.url, "--no-extract"],
        )

        assert video_to_vtt.main() == 0

        assert _Flaky503.failures_left == 0
        assert output_file.read_text().startswith("WEBVTT")
    finally:
        server.stop()


def test_max_cue_duration_is_sent_unrounded(tmp_path, fake_server, monkeypatch):
    video_file = tmp_path / "test.wav"
    video_file.write_bytes(b"dummy")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "video-to-vtt.py",
            str(video_file),
            str(tmp_path / "out.vtt"),
            "--url",
            fake_server.url,
            "--no-extract",
            "--poll-interval",
            "0.01",
            "--max-cue-duration",
            "7.5",
        ],
    )

    assert video_to_vtt.main() == 0

    result_fetches = [r for r in fake_server.requests() if r["method"] == "GET"]
    assert result_fetches and all("max_cue_duration=7.5" in r["path"] for r in result_fetches)


class _Reply:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body


class _RefusingConnection:
    """A connection whose peer refused the upload and closed the socket mid-send."""

    def __init__(self, reply=None):
        self._reply = reply

    def request(self, *args, **kwargs):
        raise BrokenPipeError("peer closed the connection")

    def getresponse(self):
        if self._reply is None:
            raise ConnectionResetError("reset")
        return self._reply

    def close(self):
        pass


def test_refused_upload_reports_the_servers_reason(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    body = json.dumps({"detail": "Upload exceeds 10 MB"}).encode()
    monkeypatch.setattr(video_to_vtt, "_connect", lambda url: _RefusingConnection(_Reply(413, body)))

    with pytest.raises(RuntimeError, match=r"413.*Upload exceeds 10 MB"):
        video_to_vtt.submit_job("http://service.invalid", audio, "fi", "60")


def test_refused_upload_without_a_readable_answer_says_what_to_check(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(video_to_vtt, "_connect", lambda url: _RefusingConnection())

    with pytest.raises(RuntimeError, match=r"closed the connection during the upload.*API key"):
        video_to_vtt.submit_job("http://service.invalid", audio, "fi", "60")


def test_upload_is_not_resent_after_the_connection_was_reset(tmp_path, monkeypatch):
    """Retrying would re-send a possibly huge file up to ten times to a server that is refusing it."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    connections = []

    def connect(url):
        connections.append(url)
        return _RefusingConnection()

    monkeypatch.setattr(video_to_vtt, "_connect", connect)
    monkeypatch.setattr(video_to_vtt, "check_service", lambda *args, **kwargs: None)
    monkeypatch.setattr(video_to_vtt.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["video-to-vtt.py", str(audio), str(tmp_path / "out.vtt"), "--no-extract", "--url", "http://service.invalid"],
    )

    assert video_to_vtt.main() == 1
    assert len(connections) == 1


def test_ffmpeg_extraction_real(tmp_path):
    """Test real ffmpeg extraction (skipped if ffmpeg not on PATH)."""
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not on PATH")

    # Create a tiny WAV file
    import wave

    audio_file = tmp_path / "test.wav"
    with wave.open(str(audio_file), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b"\x00" * (44100 * 2 * 2))  # 1 second of stereo silence

    output_file = tmp_path / "extracted.wav"

    # Test extraction - should extract as mono 16kHz
    video_to_vtt.extract_audio(audio_file, output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0

    # Verify it's 16kHz mono
    with wave.open(str(output_file), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2


def _has_ffmpeg() -> bool:
    """Check if ffmpeg is on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, TimeoutError):
        return False
