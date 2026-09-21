"""Combine per-chunk results into the job result, and derive progress numbers.

Chunk results are already in absolute time (the runner offsets them when it
writes them), and chunks do not overlap, so assembly is ordered concatenation.
Both functions are pure: they only read the manifest they are given.
"""

from datetime import datetime, timezone


class IncompleteJobError(Exception):
    """A complete result was requested but some chunks have no result yet."""


def _chunk_plan(manifest):
    return manifest.get("chunks") or []


def _total_seconds(manifest):
    audio = manifest.get("audio")
    if audio and audio.get("duration_seconds") is not None:
        return float(audio["duration_seconds"])
    plan = _chunk_plan(manifest)
    return float(plan[-1]["end"]) if plan else 0.0


def assemble(manifest, chunk_results, partial=False):
    """Concatenate `chunk_results` ({chunk index: result dict}) in index order.

    With partial=False every planned chunk must be present, otherwise
    IncompleteJobError. With partial=True whatever is there is returned and
    `complete` says whether that happens to be everything. Results for
    indices outside the plan are ignored, and a None result (what
    JobStore.read_chunk gives for an unreadable file) counts as missing.
    `models` lists the distinct `model` ids the chunks carry, in chunk order.
    """
    total = len(_chunk_plan(manifest))
    present = sorted(i for i, result in chunk_results.items() if result is not None and 0 <= i < total)
    complete = total > 0 and len(present) == total

    if not partial and not complete:
        raise IncompleteJobError(f"{len(present)} of {total} chunks are done")

    results = [chunk_results[index] for index in present]
    texts = [(result.get("text") or "").strip() for result in results]
    language = next((r["language"] for r in results if r.get("language")), None)
    # A model switch mid-job leaves chunks from different models; every one used is listed, in chunk order.
    models = list(dict.fromkeys(r["model"] for r in results if r.get("model")))

    assembled = {
        "text": " ".join(text for text in texts if text),
        "language": language or (manifest.get("params") or {}).get("language"),
        "segments": [segment for result in results for segment in result.get("segments") or []],
        "complete": complete,
        "chunks_done": len(present),
        "chunks_total": total,
        "duration_seconds": _total_seconds(manifest),
    }
    if models:  # absent when no chunk recorded its model (results written before this was tracked)
        assembled["models"] = models
    return assembled


def progress(manifest, chunks_done, now=None, baseline_chunks=0):
    """Progress numbers for polling.

    `chunks_done` is the number of finished chunks, taken to be the first
    ones in order (the runner works sequentially). ETA is the wall-clock time
    since `started_at` divided by the audio seconds done, times the seconds
    left; None until something has been measured. After a resume, pass the
    number of chunks that were already done when this run began as
    `baseline_chunks`, so they do not count as work done in this run's time.
    """
    plan = _chunk_plan(manifest)
    chunks_total = len(plan)
    done = max(0, min(int(chunks_done), chunks_total))
    total_seconds = _total_seconds(manifest)
    current_seconds = float(plan[done - 1]["end"]) if done else 0.0

    finished = chunks_total > 0 and done == chunks_total
    if finished:
        percent = 100.0
    elif total_seconds > 0:
        percent = min(100.0, current_seconds / total_seconds * 100.0)
    else:
        percent = 0.0

    eta = None
    if finished:
        eta = 0.0
    elif manifest.get("started_at"):
        baseline = max(0, min(int(baseline_chunks), done))
        baseline_seconds = float(plan[baseline - 1]["end"]) if baseline else 0.0
        worked_seconds = current_seconds - baseline_seconds
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed = (now - datetime.fromisoformat(manifest["started_at"])).total_seconds()
        if worked_seconds > 0 and elapsed > 0:
            eta = elapsed / worked_seconds * (total_seconds - current_seconds)

    return {
        "progress": percent,
        "current_seconds": current_seconds,
        "total_seconds": total_seconds,
        "eta_seconds": eta,
        "chunks_done": done,
        "chunks_total": chunks_total,
    }
