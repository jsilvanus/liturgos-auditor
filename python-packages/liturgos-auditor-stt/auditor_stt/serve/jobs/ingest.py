"""Receiving a job's stripped audio from a fleet worker: PUT /v1/jobs/{id}/audio?token=...

The fleet worker has no API key. What authorises the upload is the per-job token
the runner put in the output URL it gave the fleet (kept in the job's manifest and
removed once the audio has arrived). It accepts exactly one upload for a job that
is waiting for its audio, caps the size, and checks that what arrived is a 16-bit
mono PCM WAV before it replaces anything.
"""

import asyncio
import hmac
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from .api import JOBS_NOT_CONFIGURED, _JobIdConvertor  # noqa: F401  (registers the {jobid} convertor)
from .store import JobNotFoundError
from .wav import open_pcm16_mono

router = APIRouter(prefix="/v1/jobs")


def _check_wav(path):
    with open_pcm16_mono(path) as wav:
        if wav.num_samples == 0:
            raise ValueError("empty")


@router.put("/{job_id:jobid}/audio", status_code=204)
async def put_audio(job_id: str, request: Request, token: str = ""):
    state = request.app.state
    store = state.job_store
    if store is None:
        raise HTTPException(status_code=503, detail=JOBS_NOT_CONFIGURED)
    try:
        manifest = await asyncio.to_thread(store.load, job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="No such job") from None
    source = manifest.get("source") or {}
    expected = source.get("ingest_token") or ""
    # The same answer for a wrong token and for a job that is not waiting for audio.
    if not expected or not hmac.compare_digest(expected.encode(), token.encode()):
        raise HTTPException(status_code=403, detail="Not allowed")
    if manifest.get("status") not in ("queued", "running") or manifest.get("cancel_requested"):
        raise HTTPException(status_code=409, detail="The job is not waiting for audio")

    target = store.audio_path(job_id)
    partial = target.with_name(target.name + ".ingest")
    limit = state.max_ingest_bytes
    size = 0
    try:
        with open(partial, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"Audio is too large (limit is {limit / (1024 * 1024):g} MB)")
                await asyncio.to_thread(out.write, chunk)
        try:
            await asyncio.to_thread(_check_wav, partial)
        except (OSError, ValueError):
            raise HTTPException(status_code=422, detail="Expected a 16-bit PCM mono WAV") from None
        await asyncio.to_thread(os.replace, partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return Response(status_code=204)
