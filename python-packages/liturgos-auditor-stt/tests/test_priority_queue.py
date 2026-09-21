import asyncio
import threading
import time

import pytest

from auditor_stt.serve.queue import LIVE, InferenceQueue, QueueFullError

ZERO = {"live_depth": 0, "batch_depth": 0}


def _note(order, name):
    order.append(name)
    return name


def _blocked(order, name, gate):
    """Occupy the worker thread until the test opens the gate."""
    order.append(f"{name}:start")
    assert gate.wait(timeout=5), "test never released the gate"
    order.append(f"{name}:end")
    return name


async def _until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.005)


def _depths(queue):
    stats = queue.stats()
    return {"live_depth": stats["live_depth"], "batch_depth": stats["batch_depth"]}


@pytest.mark.asyncio
async def test_live_call_jumps_ahead_of_queued_batch_calls():
    queue = InferenceQueue(max_queue=8)
    order, gate = [], threading.Event()

    running = asyncio.create_task(queue.run_batch(_blocked, order, "running", gate))
    await _until(lambda: "running:start" in order)
    batch_1 = asyncio.create_task(queue.run_batch(_note, order, "batch-1"))
    batch_2 = asyncio.create_task(queue.run_batch(_note, order, "batch-2"))
    live = asyncio.create_task(queue.run(_note, order, "live"))
    await _until(lambda: _depths(queue) == {"live_depth": 1, "batch_depth": 3})

    gate.set()
    await asyncio.gather(running, batch_1, batch_2, live)

    assert order == ["running:start", "running:end", "live", "batch-1", "batch-2"]


@pytest.mark.asyncio
async def test_running_batch_call_is_not_interrupted_by_live_call():
    queue = InferenceQueue(max_queue=8)
    order, gate = [], threading.Event()

    batch = asyncio.create_task(queue.run_batch(_blocked, order, "batch", gate))
    await _until(lambda: "batch:start" in order)
    live = asyncio.create_task(queue.run(_note, order, "live"))
    await _until(lambda: queue.stats()["live_depth"] == 1)
    await asyncio.sleep(0.05)
    assert order == ["batch:start"]  # live waits for the running call instead of overtaking it

    gate.set()
    await asyncio.gather(batch, live)

    assert order == ["batch:start", "batch:end", "live"]


@pytest.mark.asyncio
async def test_fifo_within_each_priority():
    queue = InferenceQueue(max_queue=8)
    order, gate = [], threading.Event()

    running = asyncio.create_task(queue.run(_blocked, order, "running", gate))
    await _until(lambda: "running:start" in order)
    tasks = []
    for name, submit in [
        ("batch-1", queue.run_batch),
        ("live-1", queue.run),
        ("batch-2", queue.run_batch),
        ("live-2", queue.run),
        ("batch-3", queue.run_batch),
        ("live-3", queue.run),
    ]:
        tasks.append(asyncio.create_task(submit(_note, order, name)))
        await _until(lambda: sum(_depths(queue).values()) == len(tasks) + 1)

    gate.set()
    await asyncio.gather(running, *tasks)

    assert order[2:] == ["live-1", "live-2", "live-3", "batch-1", "batch-2", "batch-3"]


@pytest.mark.asyncio
async def test_batch_call_runs_immediately_when_idle():
    queue = InferenceQueue(max_queue=1)
    assert await queue.run_batch(_note, [], "solo") == "solo"


@pytest.mark.asyncio
async def test_run_batch_passes_arguments_and_returns_result():
    def add(a, b=0):
        return a + b

    queue = InferenceQueue()
    assert await queue.run_batch(add, 1, b=2) == 3
    assert await queue.run(add, 4, b=5) == 9


@pytest.mark.asyncio
async def test_queue_full_applies_to_live_calls_only():
    queue = InferenceQueue(max_queue=1)
    order, gate = [], threading.Event()

    live = asyncio.create_task(queue.run(_blocked, order, "live", gate))
    await _until(lambda: "live:start" in order)

    with pytest.raises(QueueFullError):
        await queue.run(_note, order, "rejected")

    # Batch calls are never rejected for depth, however many pile up.
    batches = [asyncio.create_task(queue.run_batch(_note, order, f"batch-{i}")) for i in range(20)]
    await _until(lambda: queue.stats()["batch_depth"] == 20)
    assert queue.stats()["live_depth"] == 1

    gate.set()
    await asyncio.gather(live, *batches)

    assert "rejected" not in order
    assert order[2:] == [f"batch-{i}" for i in range(20)]


@pytest.mark.asyncio
async def test_running_batch_call_does_not_use_up_live_capacity():
    queue = InferenceQueue(max_queue=1)
    order, gate = [], threading.Event()

    batch = asyncio.create_task(queue.run_batch(_blocked, order, "batch", gate))
    await _until(lambda: "batch:start" in order)
    live = asyncio.create_task(queue.run(_note, order, "live"))
    await _until(lambda: queue.stats()["live_depth"] == 1)

    with pytest.raises(QueueFullError):  # the one live slot is taken, by the waiting live call
        await queue.run(_note, order, "rejected")

    gate.set()
    await asyncio.gather(batch, live)

    assert order == ["batch:start", "batch:end", "live"]


@pytest.mark.asyncio
async def test_cancelled_waiters_are_dropped_without_wedging_the_queue():
    queue = InferenceQueue(max_queue=8)
    order, gate = [], threading.Event()

    running = asyncio.create_task(queue.run(_blocked, order, "running", gate))
    await _until(lambda: "running:start" in order)
    cancelled_live = asyncio.create_task(queue.run(_note, order, "cancelled-live"))
    survivor = asyncio.create_task(queue.run(_note, order, "survivor"))
    cancelled_batch = asyncio.create_task(queue.run_batch(_note, order, "cancelled-batch"))
    await _until(lambda: _depths(queue) == {"live_depth": 3, "batch_depth": 1})

    cancelled_live.cancel()
    cancelled_batch.cancel()
    for task in (cancelled_live, cancelled_batch):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert _depths(queue) == {"live_depth": 2, "batch_depth": 0}

    gate.set()
    assert await survivor == "survivor"
    await running

    assert order == ["running:start", "running:end", "survivor"]
    assert _depths(queue) == ZERO


@pytest.mark.asyncio
async def test_timed_out_waiter_does_not_leak_depth():
    queue = InferenceQueue(max_queue=8)
    order, gate = [], threading.Event()

    running = asyncio.create_task(queue.run(_blocked, order, "running", gate))
    await _until(lambda: "running:start" in order)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.run(_note, order, "late"), timeout=0.05)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.run_batch(_note, order, "late-batch"), timeout=0.05)
    assert _depths(queue) == {"live_depth": 1, "batch_depth": 0}

    gate.set()
    await running
    assert await queue.run(_note, order, "after") == "after"

    assert order == ["running:start", "running:end", "after"]
    assert _depths(queue) == ZERO


@pytest.mark.asyncio
async def test_cancelled_running_call_keeps_the_worker_until_its_thread_finishes():
    # A thread cannot be interrupted; letting the next call start would enter the model twice.
    queue = InferenceQueue(max_queue=8)
    order, gate = [], threading.Event()

    abandoned = asyncio.create_task(queue.run(_blocked, order, "abandoned", gate))
    await _until(lambda: "abandoned:start" in order)
    waiting = asyncio.create_task(queue.run(_note, order, "next"))
    await _until(lambda: queue.stats()["live_depth"] == 2)

    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    await asyncio.sleep(0.05)
    assert order == ["abandoned:start"]
    assert queue.stats()["live_depth"] == 2

    gate.set()
    assert await waiting == "next"

    assert order == ["abandoned:start", "abandoned:end", "next"]
    assert _depths(queue) == ZERO


@pytest.mark.asyncio
async def test_waiter_cancelled_right_after_being_granted_passes_the_worker_on():
    queue = InferenceQueue(max_queue=8)
    order = []

    queue._running = LIVE  # pretend a call holds the worker
    waiter = asyncio.create_task(queue.run(_note, order, "never-runs"))
    await _until(lambda: queue.stats()["live_depth"] == 2)

    queue._release()  # the holder finishes and hands the worker to the waiter...
    waiter.cancel()  # ...which is cancelled before it wakes up to use it
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert _depths(queue) == ZERO
    assert await queue.run(_note, order, "after") == "after"
    assert order == ["after"]


@pytest.mark.asyncio
async def test_release_skips_a_waiter_cancelled_but_not_yet_unwound():
    queue = InferenceQueue(max_queue=8)
    order = []

    queue._running = LIVE  # pretend a call holds the worker
    doomed = asyncio.create_task(queue.run(_note, order, "doomed"))
    survivor = asyncio.create_task(queue.run(_note, order, "survivor"))
    await _until(lambda: queue.stats()["live_depth"] == 3)

    doomed.cancel()
    queue._release()  # the holder finishes before the doomed task has processed its cancellation
    with pytest.raises(asyncio.CancelledError):
        await doomed

    assert await survivor == "survivor"
    assert order == ["survivor"]
    assert _depths(queue) == ZERO


@pytest.mark.asyncio
async def test_exception_in_call_propagates_and_frees_the_worker():
    def boom():
        raise ValueError("boom")

    queue = InferenceQueue(max_queue=1)
    with pytest.raises(ValueError):
        await queue.run(boom)
    with pytest.raises(ValueError):
        await queue.run_batch(boom)

    assert await queue.run(_note, [], "after") == "after"
    assert _depths(queue) == ZERO


@pytest.mark.asyncio
async def test_stats_report_live_and_batch_depth():
    queue = InferenceQueue(max_queue=3)
    assert queue.stats() == {"live_depth": 0, "batch_depth": 0, "max_queue": 3}

    order, gate = [], threading.Event()
    running = asyncio.create_task(queue.run(_blocked, order, "running", gate))
    await _until(lambda: "running:start" in order)
    assert queue.stats() == {"live_depth": 1, "batch_depth": 0, "max_queue": 3}

    tasks = [
        asyncio.create_task(queue.run(_note, order, "live")),
        asyncio.create_task(queue.run_batch(_note, order, "batch-1")),
        asyncio.create_task(queue.run_batch(_note, order, "batch-2")),
    ]
    await _until(lambda: queue.stats() == {"live_depth": 2, "batch_depth": 2, "max_queue": 3})

    gate.set()
    await asyncio.gather(running, *tasks)
    assert queue.stats() == {"live_depth": 0, "batch_depth": 0, "max_queue": 3}
