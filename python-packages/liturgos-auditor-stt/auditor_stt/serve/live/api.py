"""HTTP routes of live sessions, behind the API key."""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .session import LiveError

router = APIRouter(prefix="/v1/live", tags=["live"])

HEARTBEAT_SECONDS = 15.0


class StartRequest(BaseModel):
    source: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    client_ref: Optional[str] = None


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
async def start(body: StartRequest, request: Request):
    manager = _manager(request)
    try:
        session = manager.create(
            body.source, body.language or request.app.state.default_language, body.prompt, body.client_ref
        )
    except LiveError as exc:
        raise HTTPException(exc.status, exc.message) from None
    return {**session.info(), "events_url": f"/v1/live/{session.id}/events"}


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
