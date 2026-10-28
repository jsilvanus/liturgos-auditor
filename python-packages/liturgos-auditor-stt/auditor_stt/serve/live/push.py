"""Push delivery of live transcripts: the auditor POSTs batches to a receiver's URL.

Body: {"session_id": "<id>", "events": [<transcript objects, same as the SSE data>]}
with `Authorization: Bearer <token>` when a token was given. Events are sent in
order, up to BATCH_SIZE per request, flushed within LINGER_SECONDS. Network errors,
5xx, 429, 401 and 403 are retried with exponential backoff (capped); any other
non-2xx answer drops that batch. The in-memory queue is bounded (QUEUE_LIMIT,
oldest dropped first). The bearer token is only ever held here and written to the
Authorization header: it is never logged, echoed in errors or exposed by info().
"""

import asyncio
import fnmatch
import logging
import time
from collections import deque
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
LINGER_SECONDS = 1.0
QUEUE_LIMIT = 200
REQUEST_TIMEOUT = 10.0
BACKOFF_START = 0.5
BACKOFF_CAP = 5.0
CLOSE_SECONDS = 3.0  # best-effort flush window when the session ends
MAX_URL_LENGTH = 2048
MAX_TOKEN_LENGTH = 4096
RETRY_STATUSES = frozenset({401, 403, 429})


def parse_delivery(delivery, push_hosts, error):
    """Validate a StartRequest `delivery` object.

    Returns None for SSE-only (the default) or a dict {mode, url, token, host}.
    `error(message)` must build the exception to raise (messages never echo the URL or token).
    """
    if delivery is None:
        return None
    if not isinstance(delivery, dict):
        raise error("delivery must be an object")
    unknown = set(delivery) - {"mode", "url", "token"}
    if unknown:
        raise error("delivery accepts only mode, url and token")
    mode = delivery.get("mode", "sse")
    if mode not in ("sse", "push"):
        raise error("delivery.mode must be 'sse' or 'push'")
    url, token = delivery.get("url"), delivery.get("token")
    if mode == "sse":
        if url is not None or token is not None:
            raise error("delivery.url and delivery.token are only valid with mode 'push'")
        return None
    if not push_hosts:
        raise error("Push delivery is not enabled on this server (set AUDITOR_STT_LIVE_PUSH_HOSTS)")
    if not isinstance(url, str) or not url:
        raise error("delivery.url is required for mode 'push'")
    if len(url) > MAX_URL_LENGTH:
        raise error("delivery.url is too long")
    if token is not None:
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_LENGTH:
            raise error("delivery.token must be a non-empty string")
        if not token.isprintable() or token != token.strip():
            raise error("delivery.token contains invalid characters")
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        parts.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        raise error("delivery.url is not a valid URL") from None
    if parts.scheme.lower() not in ("http", "https") or not host:
        raise error("delivery.url must be an http:// or https:// URL")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise error("delivery.url must not contain credentials")
    if not any(fnmatch.fnmatch(host, pattern) for pattern in push_hosts):
        raise error("delivery.url host is not allowed")
    return {"mode": "push", "url": url, "token": token, "host": host}


class PushDelivery:
    """Bounded, ordered, retrying uploader for one session. Runs as an asyncio task."""

    def __init__(self, session_id, url, token, host, *, transport=None):
        self.session_id = session_id
        self.host = host
        self._url = url
        self._token = token
        self._transport = transport
        self._queue = deque()
        self._wake = asyncio.Event()
        self._closing = False
        self._deadline = None
        self._task = None
        self._warned_auth = False
        self.pushed = 0
        self.failed = 0
        self.dropped = 0

    def stats(self):
        return {"pushed": self.pushed, "failed": self.failed, "dropped": self.dropped}

    def start(self):
        self._task = asyncio.create_task(self._run())

    def submit(self, event):
        """Queue one transcript object; the oldest is dropped when the queue is full."""
        if self._closing:
            return
        if len(self._queue) >= QUEUE_LIMIT:
            self._queue.popleft()
            self.dropped += 1
        self._queue.append(event)
        self._wake.set()

    async def close(self):
        """Flush what is queued, best effort within CLOSE_SECONDS, then stop."""
        if self._task is None:
            return
        if not self._closing:
            self._closing = True
            self._deadline = time.monotonic() + CLOSE_SECONDS
            self._wake.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), CLOSE_SECONDS + 1.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            self._task.cancel()
        leftover = len(self._queue)
        if leftover:
            self._queue.clear()
            self.failed += leftover

    # --- worker ----------------------------------------------------------

    def _expired(self):
        return self._closing and time.monotonic() >= self._deadline

    async def _run(self):
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT, follow_redirects=False, transport=self._transport, headers=headers
        ) as client:
            while True:
                if not self._queue:
                    if self._closing:
                        return
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                await self._linger()
                batch = [self._queue.popleft() for _ in range(min(BATCH_SIZE, len(self._queue)))]
                await self._send(client, batch)
                if self._expired():
                    return

    async def _linger(self):
        """Give a short batch up to LINGER_SECONDS to fill."""
        end = time.monotonic() + LINGER_SECONDS
        while len(self._queue) < BATCH_SIZE and not self._closing:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), remaining)
            except asyncio.TimeoutError:
                return

    async def _send(self, client, batch):
        body = {"session_id": self.session_id, "events": batch}
        backoff = BACKOFF_START
        while True:
            timeout = REQUEST_TIMEOUT
            if self._closing:
                timeout = min(timeout, max(self._deadline - time.monotonic(), 0.1))
            reason = None
            try:
                response = await client.post(self._url, json=body, timeout=timeout)
                status = response.status_code
            except Exception as exc:  # noqa: BLE001 - never log the message: it may carry the URL
                status, reason = None, type(exc).__name__
            if status is not None and 200 <= status < 300:
                self.pushed += len(batch)
                return
            retryable = status is None or status >= 500 or status in RETRY_STATUSES
            if not retryable:
                logger.warning("Push to %s for session %s rejected (HTTP %s); batch dropped", self.host, self.session_id, status)
                self.failed += len(batch)
                return
            if status in (401, 403) and not self._warned_auth:
                self._warned_auth = True
                logger.warning("Push to %s for session %s refused (HTTP %s); retrying", self.host, self.session_id, status)
            elif status is None or status >= 500:
                logger.debug("Push to %s for session %s failed (%s); retrying", self.host, self.session_id, reason or status)
            if self._closing:  # one best-effort attempt while shutting down
                self.failed += len(batch)
                return
            try:
                await asyncio.wait_for(self._wake_on_close(), backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, BACKOFF_CAP)

    async def _wake_on_close(self):
        while not self._closing:
            await asyncio.sleep(0.05)
