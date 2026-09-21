"""Turn away job submissions before their body is read.

FastAPI parses a multipart body (spooling the file to disk) BEFORE it runs
dependencies, so neither the API-key check nor an upload cap inside the route
can stop a large unauthenticated upload from being received first. This ASGI
middleware answers POST /v1/jobs up front instead:

- 401 when an API key is configured and the Authorization header is wrong
  (same check as the `require_api_key` dependency, which it reuses)
- 503 when jobs are not configured
- 413 when the announced Content-Length is over the cap, or - for a body sent
  without one (chunked) - as soon as the bytes received exceed it

The cap is AUDITOR_STT_MAX_UPLOAD_MB plus a little room for the multipart
framing and form fields; the route enforces the exact file size while saving.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .auth import require_api_key
from .jobs.api import JOBS_NOT_CONFIGURED

SUBMIT_PATHS = frozenset({"/v1/jobs", "/v1/jobs/"})
FRAMING_ALLOWANCE = 1024 * 1024
TOO_LARGE = "Upload is too large (limit is {mb:g} MB)"


def _reject(status_code, detail, headers=None):
    # Closing the connection tells the server not to wait for the unread rest of the body.
    return JSONResponse({"detail": detail}, status_code=status_code, headers={"Connection": "close", **(headers or {})})


class UploadGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in SUBMIT_PATHS:
            await self.app(scope, receive, send)
            return

        state = scope["app"].state
        try:
            require_api_key(Request(scope))
        except HTTPException as exc:
            await _reject(exc.status_code, exc.detail, exc.headers)(scope, receive, send)
            return
        if state.job_store is None:
            await _reject(503, JOBS_NOT_CONFIGURED)(scope, receive, send)
            return

        cap = state.max_upload_bytes
        limit = cap + FRAMING_ALLOWANCE
        too_large = _reject(413, TOO_LARGE.format(mb=cap / (1024 * 1024)))
        announced = dict(scope["headers"]).get(b"content-length", b"")
        if announced.isdigit() and int(announced) > limit:
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
