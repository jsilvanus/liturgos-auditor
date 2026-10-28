"""HTTP routes of live sessions, behind the API key."""

import asyncio
import json
import math
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .session import LiveError
from .radio_delay import AdaptiveDelay

router = APIRouter(prefix="/v1/live", tags=["live"])

HEARTBEAT_SECONDS = 15.0
MAX_PREROLL_BYTES = 10 * 1024 * 1024  # 5 minutes at 16 kHz mono PCM16


class DelayReport(BaseModel):
    mismatch_ms: float
    tolerance_ms: float = 250.0


class DelayRequest(BaseModel):
    delay_seconds: float


class StartRequest(BaseModel):
    source: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    client_ref: Optional[str] = None
    delay_seconds: Optional[float] = None
    has_preroll: bool = False
    preroll_start_ms: Optional[float] = None


def _manager(request: Request):
    manager = getattr(request.app.state, "live", None)
    if manager is None:
        raise HTTPException(503, "Live sessions are not enabled")
    return manager


def _session(request: Request, session_id):
    session = _manager(request).get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown live session")
    return session


@router.post("", status_code=202)
async def start(request: Request):
    """Start a session using legacy JSON or multipart metadata plus optional PCM pre-roll."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    preroll = None
    if content_type == "application/json":
        try:
            payload = await request.json()
            body = StartRequest.parse_obj(payload)
        except (ValueError, ValidationError):
            raise HTTPException(422, "Invalid live-session JSON metadata") from None
        if body.has_preroll:
            raise HTTPException(422, "has_preroll=true requires multipart/form-data with a preroll part")
        if body.preroll_start_ms is not None:
            raise HTTPException(422, "preroll_start_ms requires a multipart pre-roll payload")
    elif content_type == "multipart/form-data":
        form = await request.form()
        try:
            metadata = form.get("metadata")
            if not isinstance(metadata, str):
                raise HTTPException(422, "multipart field 'metadata' must be a JSON string")
            try:
                payload = json.loads(metadata)
                body = StartRequest.parse_obj(payload)
            except (ValueError, ValidationError):
                raise HTTPException(422, "Invalid live-session multipart metadata") from None
            uploaded = form.get("preroll")
            if body.preroll_start_ms is not None and (not math.isfinite(body.preroll_start_ms) or body.preroll_start_ms < 0):
                raise HTTPException(422, "preroll_start_ms must be a finite non-negative epoch-millisecond timestamp")
            if body.has_preroll:
                if uploaded is None or not hasattr(uploaded, "read"):
                    raise HTTPException(422, "has_preroll=true requires a binary 'preroll' part")
                preroll = await uploaded.read(MAX_PREROLL_BYTES + 1)
                if not preroll:
                    raise HTTPException(422, "The preroll part must not be empty")
                if len(preroll) > MAX_PREROLL_BYTES:
                    raise HTTPException(413, f"preroll exceeds the {MAX_PREROLL_BYTES}-byte limit")
                if len(preroll) % 2:
                    raise HTTPException(422, "preroll PCM byte length must be even (S16LE)")
            elif body.preroll_start_ms is not None:
                raise HTTPException(422, "preroll_start_ms requires has_preroll=true")
            elif uploaded is not None:
                if not hasattr(uploaded, "read"):
                    raise HTTPException(422, "A preroll part requires has_preroll=true")
                unexpected = await uploaded.read(1)
                if unexpected:
                    raise HTTPException(422, "A preroll part requires has_preroll=true")
        finally:
            await form.close()
    else:
        raise HTTPException(415, "Content-Type must be application/json or multipart/form-data")

    manager = _manager(request)
    try:
        session = manager.create(
            body.source, body.language or request.app.state.default_language, body.prompt, body.client_ref,
            body.delay_seconds, preroll=preroll, preroll_start_ms=body.preroll_start_ms,
        )
    except LiveError as exc:
        raise HTTPException(exc.status, exc.message) from None
    return {**session.info(), "events_url": f"/v1/live/{session.id}/events"}


@router.post("/{session_id}/radio-delay/report")
async def report_radio_sync(session_id: str, body: DelayReport, request: Request):
    """Accept measured caption/audio skew and return the bounded target delay.

    The relay remains responsible for applying this target to its audio buffer.
    """
    session = _session(request, session_id)
    if not hasattr(session, "radio_delay_controller"):
        session.radio_delay_controller = AdaptiveDelay(
            minimum_seconds=0.0,
            maximum_seconds=_manager(request).config.max_delay_seconds,
        )
    try:
        result = session.radio_delay_controller.report(body.mismatch_ms, body.tolerance_ms)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    session._publish("status", {"state": "radio_delay_target", **result})
    return result


@router.put("/{session_id}/radio-delay")
async def request_radio_delay(session_id: str, body: DelayRequest, request: Request):
    """Set an explicit bounded target; this does not itself manipulate audio."""
    session = _session(request, session_id)
    if not hasattr(session, "radio_delay_controller"):
        session.radio_delay_controller = AdaptiveDelay(
            minimum_seconds=0.0,
            maximum_seconds=_manager(request).config.max_delay_seconds,
        )
    try:
        result = session.radio_delay_controller.request(body.delay_seconds)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    session._publish("status", {"state": "radio_delay_target", **result})
    return result


@router.get("/{session_id}/radio-delay")
async def get_radio_delay(session_id: str, request: Request):
    """Read the current radio-delay target for the relay to apply."""
    session = _session(request, session_id)
    if not hasattr(session, "radio_delay_controller"):
        session.radio_delay_controller = AdaptiveDelay(
            minimum_seconds=0.0,
            maximum_seconds=_manager(request).config.max_delay_seconds,
        )
    return session.radio_delay_controller.snapshot()


@router.get("")
async def list_sessions(request: Request):
    return {"sessions": [s.info() for s in _manager(request).sessions.values()]}


@router.get("/{session_id}")
async def status(session_id: str, request: Request):
    return _session(request, session_id).info()


@router.delete("/{session_id}")
async def stop(session_id: str, request: Request):
    session = _session(request, session_id)
    await session.stop("stopped")
    return JSONResponse(session.info())


@router.get("/{session_id}/events")
async def events(session_id: str, request: Request, last_event_id: Optional[str] = Header(None)):
    session = _session(request, session_id)
    try:
        after = int(last_event_id) if last_event_id else None
    except ValueError:
        after = None
    queue = session.subscribe(after)

    async def stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    return
                event_id, kind, data = event
                yield f"id: {event_id}\nevent: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        finally:
            session.unsubscribe(queue)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
