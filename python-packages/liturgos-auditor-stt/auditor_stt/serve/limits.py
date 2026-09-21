"""Turn away large uploads before their body is read.

FastAPI parses a multipart body (spooling the file to disk) BEFORE it runs
dependencies, so neither the API-key check nor an upload cap inside the route
can stop a large unauthenticated upload from being received first. This ASGI
middleware answers POST /v1/jobs and the live routes up front instead:

For POST /v1/jobs:
- 401 when an API key is configured and the Authorization header is wrong
  (same check as the `require_api_key` dependency, which it reuses)
- 503 when jobs are not configured
- 413 when the announced Content-Length is over the cap, or - for a body sent
  without one (chunked) - as soon as the bytes received exceed it

The cap is AUDITOR_STT_MAX_UPLOAD_MB plus a little room for the multipart
framing and form fields; the route enforces the exact file size while saving.

For POST /inference and POST /v1/audio/transcriptions (live routes):
- 413 when the announced Content-Length is over AUDITOR_STT_MAX_LIVE_UPLOAD_MB,
  or for a body sent without one (chunked) as soon as the bytes received exceed it.

When a request is rejected due to 401 or 413 with an announced body size, the
middleware drains up to 4 MiB of the request body to allow the client to receive
the response status code instead of a raw connection reset.
"""

import os

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .auth import require_api_key
from .jobs.api import JOBS_NOT_CONFIGURED

SUBMIT_PATHS = frozenset({"/v1/jobs", "/v1/jobs/"})
LIVE_PATHS = frozenset({"/inference", "/v1/audio/transcriptions"})
FRAMING_ALLOWANCE = 1024 * 1024
TOO_LARGE = "Upload is too large (limit is {mb:g} MB)"
BODY_DRAIN_LIMIT = 4 * 1024 * 1024  # 4 MiB: drain up to this amount to let client receive the status


def _reject(status_code, detail, headers=None):
    # Closing the connection tells the server not to wait for the unread rest of the body.
    return JSONResponse({"detail": detail}, status_code=status_code, headers={"Connection": "close", **(headers or {})})


async def _drain_body(receive, limit=BODY_DRAIN_LIMIT):
    """Read and discard body chunks up to limit to allow client to receive the rejection status."""
    drained = 0
    while drained < limit:
        message = await receive()
        if message["type"] == "http.request":
            chunk = message.get("body", b"")
            drained += len(chunk)
            if not message.get("more_body", False):
                break
        elif message["type"] == "http.disconnect":
            break


class UploadGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        path = scope["path"]

        # Handle job submission uploads with authentication and 503 check.
        if path in SUBMIT_PATHS:
            await self._handle_job_submission(scope, receive, send)
            return

        # Handle live route uploads with size cap only.
        if path in LIVE_PATHS:
            await self._handle_live_upload(scope, receive, send)
            return

        await self.app(scope, receive, send)

    async def _handle_job_submission(self, scope, receive, send):
        state = scope["app"].state
        try:
            require_api_key(Request(scope))
        except HTTPException as exc:
            announced = dict(scope["headers"]).get(b"content-length", b"")
            # Only drain body for moderate announced sizes; for huge sizes, just reject to avoid reading large bodies
            if announced.isdigit() and int(announced) <= BODY_DRAIN_LIMIT:
                await _drain_body(receive)
            await _reject(exc.status_code, exc.detail, exc.headers)(scope, receive, send)
            return
        if state.job_store is None:
            announced = dict(scope["headers"]).get(b"content-length", b"")
            # Only drain body for moderate announced sizes; for huge sizes, just reject to avoid reading large bodies
            if announced.isdigit() and int(announced) <= BODY_DRAIN_LIMIT:
                await _drain_body(receive)
            await _reject(503, JOBS_NOT_CONFIGURED)(scope, receive, send)
            return

        cap = state.max_upload_bytes
        limit = cap + FRAMING_ALLOWANCE
        too_large = _reject(413, TOO_LARGE.format(mb=cap / (1024 * 1024)))
        announced = dict(scope["headers"]).get(b"content-length", b"")
        if announced.isdigit() and int(announced) > limit:
            # Only drain body for moderate announced sizes; for huge sizes, just reject to avoid reading large bodies
            if int(announced) <= BODY_DRAIN_LIMIT:
                await _drain_body(receive)
            await too_large(scope, receive, send)
            return

        received = 0
        refused = False

        async def counting_receive():
            nonlocal received, refused
            message = await receive()
            if message["type"] == "http.request" and not refused:
                received += len(message.get("body", b""))
                if received > limit:
                    refused = True
                    await too_large(scope, receive, send)
                    # The app sees the client "leave", stops reading, and anything it
                    # then tries to send is dropped below.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if not refused:
                await send(message)

        await self.app(scope, counting_receive, guarded_send)

    async def _handle_live_upload(self, scope, receive, send):
        state = scope["app"].state
        max_live_upload_mb = float(os.environ.get("AUDITOR_STT_MAX_LIVE_UPLOAD_MB", "64"))
        cap = int(max_live_upload_mb * 1024 * 1024)
        too_large = _reject(413, TOO_LARGE.format(mb=max_live_upload_mb))

        announced = dict(scope["headers"]).get(b"content-length", b"")
        if announced.isdigit() and int(announced) > cap:
            # Only drain body for moderate announced sizes; for huge sizes, just reject to avoid reading large bodies
            if int(announced) <= BODY_DRAIN_LIMIT:
                await _drain_body(receive)
            await too_large(scope, receive, send)
            return

        received = 0
        refused = False

        async def counting_receive():
            nonlocal received, refused
            message = await receive()
            if message["type"] == "http.request" and not refused:
                received += len(message.get("body", b""))
                if received > cap:
                    refused = True
                    await too_large(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if not refused:
                await send(message)

        await self.app(scope, counting_receive, guarded_send)
