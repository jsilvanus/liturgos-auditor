"""Two-priority, single-worker inference queue.

Chunks arrive every ~5-15s per session; faster-whisper models aren't safely
shared across concurrent calls, so calls are serialized through a single
worker. Two priorities share it:

- live (`run`): captions for a running stream. Bounded by `max_queue`; rather
  than letting a backlog build unbounded latency, live calls beyond it are
  rejected immediately (503) - the Node.js WhisperHttpAdapter already treats
  STT errors as skip-and-continue.
- batch (`run_batch`): full-file jobs, submitted one ~60 s chunk at a time. Never
  rejected, and only started when no live call is waiting, so live captions
  overtake a long job between its chunks. A running call is never interrupted.

FIFO within a priority. Waiters are plain futures and the slot is handed
straight to the next one on release, so nothing is created at construction
time (no event loop needed) and no call can slip in between release and start.
"""

import asyncio
from collections import deque

LIVE = "live"
BATCH = "batch"


class QueueFullError(Exception):
    pass


class InferenceQueue:
    def __init__(self, max_queue=8):
        self.max_queue = max_queue
        self._live_waiters = deque()
        self._batch_waiters = deque()
        self._running = None  # LIVE, BATCH or None: which priority holds the worker

    def stats(self):
        return {
            "live_depth": self._depth(LIVE),
            "batch_depth": self._depth(BATCH),
            "max_queue": self.max_queue,
        }

    async def run(self, fn, *args, **kwargs):
        if self._depth(LIVE) >= self.max_queue:
            raise QueueFullError(f"Inference queue depth exceeded ({self.max_queue})")
        return await self._run(LIVE, fn, args, kwargs)

    async def run_batch(self, fn, *args, **kwargs):
        return await self._run(BATCH, fn, args, kwargs)

    def _waiters(self, priority):
        return self._live_waiters if priority == LIVE else self._batch_waiters

    def _depth(self, priority):
        return len(self._waiters(priority)) + (1 if self._running == priority else 0)

    async def _run(self, priority, fn, args, kwargs):
        await self._acquire(priority)
        work = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        try:
            return await asyncio.shield(work)
        finally:
            if work.done():
                self._release()
            else:
                # Cancelled while the thread runs. It cannot be interrupted and the
                # model must never be entered twice, so keep the slot until it ends.
                work.add_done_callback(lambda _work: self._release())

    async def _acquire(self, priority):
        if self._running is None:
            self._running = priority
            return
        waiters = self._waiters(priority)
        waiter = asyncio.get_running_loop().create_future()
        waiters.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                # The slot was already handed to us; pass it on instead of leaking it.
                self._release()
            elif waiter in waiters:
                waiters.remove(waiter)
            raise

    def _release(self):
        for priority in (LIVE, BATCH):
            waiters = self._waiters(priority)
            while waiters:
                waiter = waiters.popleft()
                if not waiter.done():  # skip waiters cancelled but not yet unwound
                    self._running = priority
                    waiter.set_result(None)
                    return
        self._running = None
