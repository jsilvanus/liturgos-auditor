"""Live sessions: pull, segment, transcribe, publish.

One session owns a reader thread (blocking reads of the PCM stream, clock check,
segmenter) and an asyncio task that sends each segment through the inference
queue at LIVE priority and publishes the text. Segments are handed over without
blocking the reader, so a slow model never stalls the stream; a segment that has
waited longer than `max_lag_seconds` is dropped with a status event instead.

Events (all JSON, `id` = the session's event counter):
  transcript  {sequence, text, language, start, end, wall_start, wall_end}
  status      {state, ...}   starting, running, reconnecting, clock_reset, dropped, ended
  error       {message}

`start`/`end` are seconds since the session began; `wall_start`/`wall_end` are
absolute UTC (ISO 8601) taken from the clock anchor, so a consumer never needs to
know when the session started. Sessions are not durable.
"""

import asyncio
import fnmatch
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import numpy as np

from ..jobs.fleet import FleetConfig, FleetStripper, FleetUnavailableError, _scrubber
from ..jobs.pipeline import carry_prompt
from ..queue import QueueFullError
from .clock import WallClock, iso
from .segmenter import SAMPLE_RATE, Segmenter, vad_speech_ranges
from .source import LIVE_SCHEMES, FleetPcmStream, LocalPcmStream

logger = logging.getLogger(__name__)

READ_BYTES = 8000  # 0.25 s of 16 kHz mono PCM16
EVENT_HISTORY = 500
KEEP_ENDED_SECONDS = 300.0


class LiveError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class LiveConfig:
    source_hosts: list = field(default_factory=list)  # fnmatch patterns; empty disables live sessions
    max_sessions: int = 2
    requires: list = field(default_factory=list)  # fleet capabilities, e.g. ["net:mediamtx"]
    reconnect_seconds: float = 120.0
    max_lag_seconds: float = 30.0
    max_segment_seconds: float = 10.0
    pause_seconds: float = 0.6
    carry_chars: int = 200
    read_timeout: float = 120.0
    clock_drift_seconds: float = 1.0

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env

        def split(name):
            return [part.strip() for part in env.get(name, "").split(",") if part.strip()]

        config = cls(
            source_hosts=[host.lower() for host in split("AUDITOR_STT_LIVE_SOURCE_HOSTS")],
            requires=split("AUDITOR_STT_LIVE_REQUIRES"),
        )
        for name, attr, cast in (
            ("AUDITOR_STT_MAX_LIVE_SESSIONS", "max_sessions", int),
            ("AUDITOR_STT_LIVE_RECONNECT_SECONDS", "reconnect_seconds", float),
            ("AUDITOR_STT_LIVE_MAX_LAG_SECONDS", "max_lag_seconds", float),
            ("AUDITOR_STT_LIVE_MAX_SEGMENT_SECONDS", "max_segment_seconds", float),
            ("AUDITOR_STT_LIVE_PAUSE_SECONDS", "pause_seconds", float),
            ("AUDITOR_STT_LIVE_READ_TIMEOUT", "read_timeout", float),
            ("AUDITOR_STT_LIVE_CLOCK_DRIFT_SECONDS", "clock_drift_seconds", float),
        ):
            if env.get(name):
                setattr(config, attr, cast(env[name]))
        return config


def check_source(url, hosts):
    """Validate a source URL: rtsp/srt only, host on the allow-list."""
    parts = urlsplit(url or "")
    if parts.scheme.lower() not in LIVE_SCHEMES or not parts.hostname:
        raise LiveError(422, f"source must be an {'/'.join(s + '://' for s in LIVE_SCHEMES)} URL")
    if not hosts:
        raise LiveError(403, "Live sessions are not enabled (set AUDITOR_STT_LIVE_SOURCE_HOSTS)")
    host = parts.hostname.lower()
    if not any(fnmatch.fnmatch(host, pattern) for pattern in hosts):
        raise LiveError(403, "source host is not allowed")


class LiveSession:
    def __init__(self, manager, session_id, source, language, prompt, client_ref):
        self.manager = manager
        self.id = session_id
        self.source = source
        self.language = language
        self.prompt = prompt
        self.client_ref = client_ref
        self.state = "starting"
        self.reason = None
        self.created = manager.now()
        self.ended_at = None
        self.transcripts = 0
        self.attempt = 0
        self.events = deque(maxlen=EVENT_HISTORY)
        self._counter = 0
        self._subscribers = []
        self._stop = threading.Event()
        self._stream = None
        self._queue = None
        self._loop = None
        self._tasks = []
        self._scrub = _scrubber(source)
        self._first_wall = None

    # --- public ----------------------------------------------------------

    def info(self):
        return {
            "id": self.id,
            "status": self.state,
            "reason": self.reason,
            "client_ref": self.client_ref,
            "language": self.language,
            "transcripts": self.transcripts,
            "attempts": self.attempt,
            "created": iso(self.created),
            "ended": iso(self.ended_at) if self.ended_at else None,
        }

    @property
    def finished(self):
        return self.state == "ended"

    def subscribe(self, last_event_id=None):
        """An asyncio.Queue of (id, type, data); replays events after `last_event_id`, None marks the end."""
        queue = asyncio.Queue()
        for event in self.events:
            if last_event_id is None or event[0] > last_event_id:
                queue.put_nowait(event)
        if self.finished:
            queue.put_nowait(None)
        else:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue):
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def start(self):
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._tasks = [asyncio.create_task(self._run()), asyncio.create_task(self._transcribe())]
        self._publish("status", {"state": "starting"})

    async def stop(self, reason="stopped"):
        self._stop.set()
        self._close_stream()
        if self.reason is None:
            self.reason = reason
        for task in self._tasks:
            if task is not asyncio.current_task():
                task.cancel()
        for task in self._tasks:
            if task is not asyncio.current_task():
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        self._end()

    # --- events ----------------------------------------------------------

    def _publish(self, kind, data):
        self._counter += 1
        event = (self._counter, kind, data)
        self.events.append(event)
        for queue in list(self._subscribers):
            queue.put_nowait(event)

    def _post(self, fn, *args):
        """Run `fn` on the event loop from the reader thread."""
        try:
            self._loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # loop closed during shutdown
            pass

    def _end(self):
        if self.finished:
            return
        self.state = "ended"
        self.ended_at = self.manager.now()
        self._publish("status", {"state": "ended", "reason": self.reason or "stopped"})
        for queue in self._subscribers:
            queue.put_nowait(None)
        self._subscribers.clear()

    # --- the stream ------------------------------------------------------

    def _close_stream(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                logger.warning("Closing the live stream of %s failed", self.id)

    async def _run(self):
        deadline_start = None
        backoff = 1.0
        try:
            while not self._stop.is_set():
                self.attempt += 1
                received = await asyncio.to_thread(self._read_attempt)
                if self._stop.is_set():
                    break
                if received:
                    deadline_start, backoff = None, 1.0
                if deadline_start is None:
                    deadline_start = time.monotonic()
                if time.monotonic() - deadline_start >= self.manager.config.reconnect_seconds:
                    self.reason = "source_lost"
                    self._publish("error", {"message": "the stream could not be reached again"})
                    break
                self.state = "reconnecting"
                self._publish("status", {"state": "reconnecting", "attempt": self.attempt})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Live session %s failed", self.id)
            self.reason = "error"
            self._publish("error", {"message": self._scrub(f"{type(exc).__name__}")})
        finally:
            await self._queue.put(None)
        # Let the transcriber finish what it has, then close the session.
        await asyncio.shield(self._tasks[1])
        self._end()

    def _read_attempt(self):
        """Blocking: one connection to the stream. Returns the number of samples received."""
        clock = WallClock(SAMPLE_RATE, now=self.manager.now, drift_limit=self.manager.config.clock_drift_seconds)
        segmenter = Segmenter(
            SAMPLE_RATE,
            max_seconds=self.manager.config.max_segment_seconds,
            pause_seconds=self.manager.config.pause_seconds,
            speech_ranges=self.manager.speech_ranges,
        )
        try:
            stream = self.manager.open_stream(self.source, self.id, self.attempt)
        except Exception as exc:  # noqa: BLE001 - reported, then retried until the reconnect window ends
            self._post(self._publish, "error", {"message": self._scrub(str(exc) or type(exc).__name__)})
            return 0
        self._stream = stream
        if self._stop.is_set():
            self._close_stream()
            return 0
        total = 0
        try:
            while not self._stop.is_set():
                data = stream.read(READ_BYTES)
                if not data:
                    break
                data = data[: len(data) // 2 * 2]
                samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                total += len(samples)
                if total == len(samples):
                    self._post(self._set_running)
                drift = clock.check(total, caught_up=len(data) == READ_BYTES and self._queue.qsize() == 0)
                if drift is not None:
                    self._post(self._publish, "status", {"state": "clock_reset", "drift_seconds": round(drift, 3)})
                for segment in segmenter.feed(samples):
                    self._hand_over(clock, segment)
            tail = segmenter.flush() if not self._stop.is_set() else None
            if tail is not None:
                self._hand_over(clock, tail)
        finally:
            self._close_stream()
        return total

    def _set_running(self):
        if self.state != "ended":
            self.state = "running"
            self._publish("status", {"state": "running", "attempt": self.attempt})

    def _hand_over(self, clock, segment):
        wall_start = clock.time_of(segment.start_sample)
        wall_end = clock.time_of(segment.end_sample)
        self._post(self._queue.put_nowait, (self.manager.now(), segment.samples, wall_start, wall_end))

    # --- the model -------------------------------------------------------

    async def _transcribe(self):
        previous = ""
        manager = self.manager
        while True:
            item = await self._queue.get()
            if item is None:
                return
            queued, samples, wall_start, wall_end = item
            if manager.now() - queued > manager.config.max_lag_seconds:
                self._publish("status", {"state": "dropped", "wall_start": iso(wall_start), "wall_end": iso(wall_end)})
                continue
            host = manager.host_getter()
            try:
                result = await manager.queue.run(
                    host.transcribe_array,
                    samples,
                    self.language,
                    prompt=carry_prompt(self.prompt, previous, manager.config.carry_chars),
                    vad=False,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                )
            except QueueFullError:
                self._publish("status", {"state": "dropped", "reason": "busy", "wall_start": iso(wall_start), "wall_end": iso(wall_end)})
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad segment must not end the session
                logger.warning("Live segment failed: %s", type(exc).__name__)
                self._publish("error", {"message": f"transcription failed ({type(exc).__name__})"})
                continue
            text = (result.get("text") or "").strip()
            if not text:
                continue
            previous = f"{previous} {text}"[-1000:]
            self.transcripts += 1
            self._publish(
                "transcript",
                {
                    "sequence": self.transcripts,
                    "text": text,
                    "language": result.get("language") or self.language,
                    "start": round(wall_start - self.created, 3),
                    "end": round(wall_end - self.created, 3),
                    "wall_start": iso(wall_start),
                    "wall_end": iso(wall_end),
                },
            )


class LiveManager:
    def __init__(
        self,
        config,
        host_getter,
        queue,
        *,
        fleet=None,
        fleet_fallback=True,
        stream_factory=None,
        speech_ranges=None,
        now=time.time,
    ):
        self.config = config
        self.host_getter = host_getter
        self.queue = queue
        self.fleet = fleet
        self.fleet_fallback = fleet_fallback
        self._stream_factory = stream_factory
        self.speech_ranges = speech_ranges or vad_speech_ranges
        self.now = now
        self.sessions = {}

    # --- sessions --------------------------------------------------------

    def create(self, source, language, prompt=None, client_ref=None):
        check_source(source, self.config.source_hosts)
        self._purge()
        active = [s for s in self.sessions.values() if not s.finished]
        if len(active) >= self.config.max_sessions:
            raise LiveError(429, f"At most {self.config.max_sessions} live sessions at a time")
        session = LiveSession(self, uuid.uuid4().hex[:16], source, language, prompt, client_ref)
        self.sessions[session.id] = session
        session.start()
        return session

    def get(self, session_id):
        return self.sessions.get(session_id)

    def _purge(self):
        cutoff = self.now() - KEEP_ENDED_SECONDS
        for key in [k for k, s in self.sessions.items() if s.finished and s.ended_at < cutoff]:
            del self.sessions[key]

    async def shutdown(self):
        for session in list(self.sessions.values()):
            if not session.finished:
                await session.stop("shutdown")

    # --- sources ---------------------------------------------------------

    def open_stream(self, source, session_id, attempt):
        """Blocking. Fleet first (when configured), the local ffmpeg when the fleet is unavailable and fallback is on."""
        if self._stream_factory is not None:
            return self._stream_factory(source, session_id, attempt)
        if self.fleet is not None:
            try:
                return FleetPcmStream(
                    self.fleet,
                    f"auditor-live-{session_id}-{attempt}",
                    source,
                    requires=self.config.requires,
                    read_timeout=self.config.read_timeout,
                )
            except FleetUnavailableError:
                if not self.fleet_fallback:
                    raise
                logger.warning("Fleet unavailable for live session %s; running ffmpeg locally", session_id)
        return LocalPcmStream(source)
