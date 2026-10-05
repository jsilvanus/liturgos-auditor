"""FastAPI inference server: whisper.cpp-compatible /inference, OpenAI-compatible
/v1/audio/transcriptions, /health and /status, and the batch job API (/v1/jobs).

The Node.js WhisperHttpAdapter (packages/plugins/lcyt-rtmp/src/stt-adapters/
whisper-http.js) already speaks the /inference protocol, so pointing
WHISPER_HTTP_URL at this service requires no changes on that side. lcyt's
OpenAiAdapter can use the /v1 route (and an API key) the same way.

Jobs (full-file transcription for saarnavideo) are enabled by AUDITOR_STT_DATA_DIR;
without it /v1/jobs answers 503 and nothing is written to disk.

The model comes from AUDITOR_STT_MODEL: a faster-whisper alias, a path, or
`registry:<name>[@<version>]` (serve/registry.py). `GET /model` shows what is
loaded and what the registry holds; `POST /model` switches models without a
restart. The new model is loaded off-thread while the old one keeps serving and
replaces it only once loaded, so a failed load changes nothing and a request
already running finishes on the model it started with. GPU memory: old and new
model are resident together during a switch (the old one is freed when the last
request using it ends), so the device needs room for both; on device=auto a load
that runs out of GPU memory falls back to the CPU, which the log reports as a
device change. The endpoint is off unless an API key is configured and only
accepts `registry:` specs and AUDITOR_STT_ALLOWED_MODELS entries, never paths.
"""

import asyncio
import gc
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .audio import temp_audio_file
from .auth import require_api_key
from .jobs.api import router as jobs_router
from .jobs.runner import JobRunner
from .jobs.store import JobStore
from .limits import UploadGuard
from .model import AudioDecodeError, ModelHost, ModelLoadError
from .queue import InferenceQueue, QueueFullError
from .registry import REGISTRY_PREFIX, RegistryError, default_registry_dir, list_models, resolve

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "large-v3-turbo"

# srt/vtt are not served on the live routes (jobs render captions), so they are rejected like any unknown format.
RESPONSE_FORMATS = ("json", "verbose_json", "text")

SWITCH_NEEDS_KEY = "Model switching requires AUDITOR_STT_API_KEY"


class ModelSwitchRequest(BaseModel):
    model: str = Field(min_length=1, max_length=512)


def default_host_factory(model_id):
    """An unloaded ModelHost with the device settings from the environment (same as at startup)."""
    return ModelHost(
        model_id=model_id,
        model_dir=os.environ.get("AUDITOR_STT_MODEL_DIR"),
        device=os.environ.get("AUDITOR_STT_DEVICE", "auto"),
        compute_type=os.environ.get("AUDITOR_STT_COMPUTE_TYPE"),
    )


def require_switch_enabled(request: Request) -> None:
    # lcyt's default deployment runs without a key behind network isolation; there an open
    # POST /model would let any peer swap the model, so switching needs a configured key.
    if not request.app.state.api_key:
        raise HTTPException(status_code=403, detail=SWITCH_NEEDS_KEY)


def _use_label(host, label):
    """Show a registry model as `registry:<name>@<version>` instead of its directory.

    ModelHost loads by `model_id`, so the label can only replace the path once loading is done.
    """
    host.model_id = label


def create_app(
    model_host: Optional[ModelHost] = None,
    queue: Optional[InferenceQueue] = None,
    api_key: Optional[str] = None,
    data_dir=None,
    media_root=None,
    job_store: Optional[JobStore] = None,
    runner: Optional[JobRunner] = None,
    host_factory=None,
    registry_dir=None,
    allowed_models=None,
) -> FastAPI:
    # `host_factory(model_id)` builds an unloaded host; tests inject stubs through it, both for
    # the startup model and for POST /model.
    host_factory = host_factory or default_host_factory
    registry_dir = Path(registry_dir) if registry_dir is not None else default_registry_dir()

    startup_label = None  # set when the startup model is a registry entry (see _use_label)
    startup_error = None
    if model_host is not None:
        host = model_host
        model_source = getattr(model_host, "model_id", None)
    else:
        model_source = os.environ.get("AUDITOR_STT_MODEL", DEFAULT_MODEL)
        try:
            path, label = resolve(model_source, registry_dir)
            host = host_factory(path)
            startup_label = label if label != path else None
        except RegistryError as exc:
            # Not fatal: an unloaded placeholder keeps /health at 503 (as a failed model load does),
            # and POST /model can still bring a working model in without a restart.
            startup_error = exc
            host = default_host_factory(model_source)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("API key auth %s", "enabled" if app.state.api_key else "disabled")
        if startup_error is not None:
            logger.error(
                "Cannot use AUDITOR_STT_MODEL=%s (%s); /health will report not-loaded until a model is switched in",
                model_source,
                startup_error,
            )
        else:
            try:
                app.state.model_host.load()
            except ModelLoadError:
                logger.exception("Model failed to load at startup; /health will report not-loaded until fixed")
            if startup_label is not None:
                _use_label(app.state.model_host, startup_label)
        # The runner waits for a loaded model itself, so a failed load above does not stop it.
        if app.state.job_runner is not None:
            await app.state.job_runner.start()
        yield
        if app.state.job_runner is not None:
            await app.state.job_runner.stop()

    app = FastAPI(title="auditor-stt", version="0.1.0", lifespan=lifespan)
    app.state.model_host = host
    app.state.model_source = model_source  # the spec the current host was loaded from
    app.state.model_switching = False  # one POST /model at a time
    app.state.queue = queue or InferenceQueue(max_queue=int(os.environ.get("AUDITOR_STT_MAX_QUEUE", "8")))
    app.state.default_language = os.environ.get("AUDITOR_STT_DEFAULT_LANGUAGE", "fi")
    # The parameter wins over the environment; an empty key means no auth.
    configured_key = api_key if api_key is not None else os.environ.get("AUDITOR_STT_API_KEY", "").strip()
    app.state.api_key = configured_key or None
    # Besides `registry:` specs, POST /model accepts only these aliases (never a path, unless the operator lists it).
    if allowed_models is None:
        allowed_models = os.environ.get("AUDITOR_STT_ALLOWED_MODELS", "").split(",")
    app.state.allowed_models = frozenset(entry.strip() for entry in allowed_models if entry.strip())

    # Jobs need somewhere to keep their files. Without a data dir nothing is created on disk,
    # no runner starts, and /v1/jobs answers 503.
    data_dir = data_dir or os.environ.get("AUDITOR_STT_DATA_DIR") or None
    if job_store is None and runner is not None:
        job_store = runner.store
    if job_store is None and data_dir:
        job_store = JobStore(Path(data_dir) / "jobs")
    if job_store is not None and runner is None:
        runner = JobRunner(job_store, lambda: app.state.model_host, app.state.queue)
    app.state.job_store = job_store
    app.state.job_runner = runner
    # `source_path` submissions are only accepted under this directory; unset disables them.
    media_root = media_root or os.environ.get("AUDITOR_STT_MEDIA_ROOT") or None
    app.state.media_root = Path(media_root).resolve() if media_root else None
    app.state.max_upload_bytes = int(float(os.environ.get("AUDITOR_STT_MAX_UPLOAD_MB", "2048")) * 1024 * 1024)
    # `source_url` submissions are only fetched from these hosts (fnmatch patterns, comma separated); unset disables them.
    app.state.source_url_hosts = [
        host.strip().lower() for host in os.environ.get("AUDITOR_STT_SOURCE_URL_HOSTS", "").split(",") if host.strip()
    ]
    app.state.source_url_timeout = float(os.environ.get("AUDITOR_STT_SOURCE_URL_TIMEOUT", "300"))
    app.add_middleware(UploadGuard)

    protected = [Depends(require_api_key)]

    def model_status():
        host = app.state.model_host
        return {
            "status": "ok" if host.loaded else "loading",
            "model_id": host.model_id,
            "device": host.device,
            "compute_type": host.compute_type,
            "loaded": host.loaded,
        }

    def check_response_format(response_format):
        if response_format not in RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response_format '{response_format}' (supported: {', '.join(RESPONSE_FORMATS)})",
            )

    async def transcribe_upload(file, language, **supplied):
        host = app.state.model_host
        if not host.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        # Forward only what the client sent, so hosts that don't know an option are never handed it.
        options = {name: value for name, value in supplied.items() if value is not None}

        data = await file.read()
        lang = language or app.state.default_language

        try:
            with temp_audio_file(data, filename=file.filename, content_type=file.content_type) as path:
                return await app.state.queue.run(host.transcribe, path, language=lang, **options)
        except QueueFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AudioDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/health")
    async def health():
        body = model_status()
        if not body["loaded"]:
            return JSONResponse(body, status_code=503)
        return body

    @app.get("/status", dependencies=protected)
    async def status():
        return {"queue": app.state.queue.stats(), "model": model_status()}

    def model_info():
        host = app.state.model_host
        return {
            "model_id": host.model_id,
            "device": host.device,
            "compute_type": host.compute_type,
            "loaded": host.loaded,
            "source": app.state.model_source,
        }

    @app.get("/model", dependencies=protected)
    async def get_model():
        return {"model": model_info(), "registry": await asyncio.to_thread(list_models, registry_dir)}

    def load_new_host(path):
        new_host = host_factory(path)
        new_host.load()
        if not new_host.loaded:
            raise ModelLoadError("Host reports not loaded after load()")
        return new_host

    async def swap_model(spec, path, label):
        """Load `spec` beside the running host and swap it in; runs as its own task (see switch_model)."""
        try:
            try:
                new_host = await asyncio.to_thread(load_new_host, path)
            except Exception:  # noqa: BLE001 - whatever a load raised, the running model stays
                logger.exception("Switching to %s failed; the current model keeps serving", label)
                raise HTTPException(
                    status_code=500, detail=f"Could not load model '{label}'; the current model is still serving"
                ) from None
            if label != path:
                _use_label(new_host, label)
            old_host = app.state.model_host
            device_change = (old_host.device, old_host.compute_type) != (new_host.device, new_host.compute_type)
            # One synchronous step: from here on new requests get the new host, while requests that
            # already took a reference to the old one finish on it.
            app.state.model_host = new_host
            app.state.model_source = spec
            logger.info(
                "Switched model from %s to %s (%s/%s)", old_host.model_id, label, new_host.device, new_host.compute_type
            )
            if device_change:
                logger.warning(
                    "The new model runs on %s/%s, not %s/%s like the previous one",
                    new_host.device, new_host.compute_type, old_host.device, old_host.compute_type,
                )
            # The old host (and its weights) is freed once the last request using it ends.
            del old_host
            await asyncio.to_thread(gc.collect)
            return {"model": model_info()}
        finally:
            app.state.model_switching = False

    @app.post("/model", dependencies=[*protected, Depends(require_switch_enabled)])
    async def switch_model(body: ModelSwitchRequest):
        spec = body.model.strip()
        if not (spec.startswith(REGISTRY_PREFIX) or spec in app.state.allowed_models):
            raise HTTPException(
                status_code=422,
                detail=f"Model must be a {REGISTRY_PREFIX}<name>[@<version>] spec or listed in AUDITOR_STT_ALLOWED_MODELS",
            )
        try:
            path, label = await asyncio.to_thread(resolve, spec, registry_dir)
        except RegistryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        if app.state.model_switching:
            raise HTTPException(status_code=409, detail="A model switch is already in progress")
        app.state.model_switching = True
        # Its own task, so a client that gives up cannot abandon a half-done swap.
        task = asyncio.ensure_future(swap_model(spec, path, label))
        task.add_done_callback(lambda done: done.cancelled() or done.exception())  # outcome is reported below
        return await asyncio.shield(task)

    @app.post("/inference", dependencies=protected)
    async def inference(
        file: UploadFile = File(...),
        language: Optional[str] = Form(None),
        model: Optional[str] = Form(None),  # noqa: ARG001 - whisper.cpp compat; ignored (one resident model)
        prompt: Optional[str] = Form(None),
        vad: Optional[bool] = Form(None),
        temperature: Optional[float] = Form(None),
        response_format: str = Form("json"),
    ):
        check_response_format(response_format)
        result = await transcribe_upload(file, language, prompt=prompt, vad=vad, temperature=temperature)
        if response_format == "text":
            return PlainTextResponse(result["text"])
        return result

    @app.post("/v1/audio/transcriptions", dependencies=protected)
    async def openai_transcriptions(
        file: UploadFile = File(...),
        model: Optional[str] = Form(None),  # noqa: ARG001 - OpenAI clients always send it; ignored (one resident model)
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        temperature: Optional[float] = Form(None),
        response_format: str = Form("json"),
    ):
        check_response_format(response_format)
        result = await transcribe_upload(file, language, prompt=prompt, temperature=temperature)
        if response_format == "text":
            return PlainTextResponse(result["text"])
        if response_format == "verbose_json":
            return result
        return {"text": result["text"]}

    # Routers for jobs and the model registry are included here, behind the API key:
    #   app.include_router(router, dependencies=protected)
    app.include_router(jobs_router, dependencies=protected)

    return app


app = create_app()
