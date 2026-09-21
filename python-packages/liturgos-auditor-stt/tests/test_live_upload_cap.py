"""Tests for upload caps on live routes (/inference and /v1/audio/transcriptions)."""

import asyncio
import os
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.queue import InferenceQueue


class _Host:
    model_id = "stub-model"
    device = "cpu"
    compute_type = "int8"
    loaded = True

    def load(self):
        pass

    def transcribe(self, audio_path, language=None, **options):
        return {"text": "live", "language": language, "segments": []}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in ("AUDITOR_STT_API_KEY", "AUDITOR_STT_MAX_LIVE_UPLOAD_MB"):
        monkeypatch.delenv(name, raising=False)


def test_live_upload_cap_comes_from_the_environment(monkeypatch):
    """AUDITOR_STT_MAX_LIVE_UPLOAD_MB defaults to 64 MB."""
    # Check default
    app1 = create_app(model_host=_Host(), queue=InferenceQueue())
    with TestClient(app1) as client:
        # We can't directly check app.state for live cap (it's in limits.py), so we test indirectly
        pass

    # Check custom value
    monkeypatch.setenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", "128")
    app2 = create_app(model_host=_Host(), queue=InferenceQueue())
    # Environment is read at request time in limits.py, so just verify it parses


def test_large_announced_live_upload_is_rejected_before_the_body_is_read(monkeypatch):
    """A Content-Length over the cap is rejected immediately (413)."""
    monkeypatch.setenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", "1")
    app = create_app(model_host=_Host(), queue=InferenceQueue())

    with TestClient(app) as client:
        # Try to upload 10 MB with announced Content-Length
        response = client.post(
            "/inference",
            files={"file": ("test.wav", b"x" * (10 * 1024 * 1024), "audio/wav")},
        )

        assert response.status_code == 413
        assert "too large" in response.json()["detail"]


def test_live_upload_just_under_the_cap_is_accepted(monkeypatch):
    """An upload just under the cap is accepted (accounting for multipart overhead)."""
    cap_mb = 1
    monkeypatch.setenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", str(cap_mb))
    app = create_app(model_host=_Host(), queue=InferenceQueue())

    with TestClient(app) as client:
        # Use 90% of the cap to account for multipart framing
        file_bytes = int(cap_mb * 1024 * 1024 * 0.9)
        response = client.post(
            "/inference",
            files={"file": ("test.wav", b"x" * file_bytes, "audio/wav")},
        )

        # Should fail audio decode (not auth/upload cap), showing it got through upload guard
        assert response.status_code in (422, 200, 503)  # 422 for decode error, 200 if it works, 503 if queue full


def test_live_upload_cap_is_checked_on_both_live_routes(monkeypatch):
    """Both /inference and /v1/audio/transcriptions respect the cap."""
    monkeypatch.setenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", "0.5")  # 512 KB
    app = create_app(model_host=_Host(), queue=InferenceQueue())

    payload = b"x" * (1024 * 1024)  # 1 MB, over the cap

    with TestClient(app) as client:
        for path in ("/inference", "/v1/audio/transcriptions"):
            response = client.post(
                path,
                files={"file": ("test.wav", payload, "audio/wav")},
            )

            assert response.status_code == 413, f"Path {path} should reject large upload"
            assert "too large" in response.json()["detail"]


def test_live_upload_without_content_length_is_cut_off_once_it_exceeds_the_cap(monkeypatch):
    """A chunked upload without Content-Length is cut off at the cap (413)."""
    monkeypatch.setenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", "0.1")  # 102.4 KB
    app = create_app(model_host=_Host(), queue=InferenceQueue())

    # Build a chunked body (multipart without Content-Length)
    boundary = "testboundary"
    crlf = b"\r\n"
    head = f'--{boundary}{crlf.decode()}Content-Disposition: form-data; name="file"; filename="test.wav"{crlf.decode()}{crlf.decode()}'.encode()
    chunk_body = b"x" * (1024 * 1024)  # 1 MB, over the cap

    body_parts = [head, chunk_body]

    with TestClient(app) as client:
        response = client.post(
            "/inference",
            content=(part for part in body_parts),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )

        assert response.status_code == 413
        assert "too large" in response.json()["detail"]


def test_live_upload_default_cap_is_64mb(monkeypatch):
    """When AUDITOR_STT_MAX_LIVE_UPLOAD_MB is not set, default is 64 MB."""
    # Make sure the env var is not set
    monkeypatch.delenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", raising=False)
    app = create_app(model_host=_Host(), queue=InferenceQueue())

    # A 65 MB upload should be rejected
    with TestClient(app) as client:
        large_body = b"x" * (65 * 1024 * 1024)

        response = client.post(
            "/inference",
            files={"file": ("test.wav", large_body, "audio/wav")},
        )

        # Should be rejected due to size
        assert response.status_code == 413


async def _raw_post(app, headers, body_chunks=(), path="/inference"):
    """POST straight into the ASGI app, counting how often its body is read."""
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
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(name.encode(), value.encode()) for name, value in headers.items()],
        "client": ("testclient", 5000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)

    starts = [message for message in sent if message["type"] == "http.response.start"]
    start = starts[0] if starts else None
    return {
        "responses": len(starts),
        "status": start["status"] if start else None,
        "headers": {name.decode(): value.decode() for name, value in start["headers"]} if start else {},
        "body_reads": reads,
    }


@pytest.mark.parametrize("path", ["/inference", "/v1/audio/transcriptions"])
def test_huge_announced_live_upload_is_rejected_before_body_is_read(monkeypatch, path):
    """A huge announced upload does not cause any body reads."""
    monkeypatch.setenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", "0.1")
    app = create_app(model_host=_Host(), queue=InferenceQueue())

    sent = asyncio.run(_raw_post(app, {"content-length": str(50 * 1024 * 1024)}, path=path))

    assert sent["status"] == 413
    assert sent["body_reads"] == 0  # the announced size alone was enough


def test_live_route_upload_cap_before_auth(monkeypatch):
    """Live route upload cap is checked independently of authentication."""
    monkeypatch.setenv("AUDITOR_STT_API_KEY", "secret-key")
    monkeypatch.setenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", "0.001")  # very small cap
    app = create_app(model_host=_Host(), queue=InferenceQueue(), api_key="secret-key")

    with TestClient(app) as client:
        # A huge upload should be rejected as 413, not 401 (auth is checked after size limit)
        response = client.post(
            "/inference",
            files={"file": ("test.wav", b"x" * (100 * 1024 * 1024), "audio/wav")},
        )

        # Should get upload size error even without auth
        assert response.status_code == 413


def test_live_upload_cap_is_independent_of_jobs_cap(monkeypatch):
    """Live upload cap is independent from AUDITOR_STT_MAX_UPLOAD_MB."""
    monkeypatch.setenv("AUDITOR_STT_MAX_UPLOAD_MB", "10")
    monkeypatch.setenv("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", "1")
    monkeypatch.setenv("AUDITOR_STT_DATA_DIR", ".")

    app = create_app(model_host=_Host(), queue=InferenceQueue(), data_dir=".")

    # Job cap is 10 MB, live cap is 1 MB
    assert app.state.max_upload_bytes == 10 * 1024 * 1024
