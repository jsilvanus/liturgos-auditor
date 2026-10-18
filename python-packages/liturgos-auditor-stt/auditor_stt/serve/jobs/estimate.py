"""Where a queued job stands: audio ahead of it and when it should start.

Pure functions over manifests. The runner takes queued jobs oldest first, so the audio
ahead of a job is what is left of the running job plus every older queued job. A job whose
duration is not known leaves the estimate unknown instead of guessing. While live captions
run they take the model first, so every estimate is then a minimum.
"""


def job_duration(manifest):
    """Seconds of audio in a job, or None when nobody has measured it yet."""
    audio = manifest.get("audio")
    if audio and audio.get("duration_seconds") is not None:
        return float(audio["duration_seconds"])
    value = manifest.get("duration_seconds")
    return None if value is None else float(value)


def _remaining(manifest):
    total = job_duration(manifest)
    if total is None:
        return None
    return max(0.0, total - float(manifest.get("current_seconds") or 0.0))


def queue_overview(manifests, rate, live_active=False):
    """`manifests` oldest first. Returns {"summary": {...}, "jobs": {job id: queue block}}."""
    running = [m for m in manifests if m.get("status") == "running"]
    queued = [m for m in manifests if m.get("status") == "queued"]

    ahead = 0.0
    unknown = 0
    for manifest in running:
        remaining = _remaining(manifest)
        if remaining is None:
            unknown += 1
        else:
            ahead += remaining

    jobs = {}
    queued_seconds = 0.0
    unknown_queued = 0
    for position, manifest in enumerate(queued, start=1):
        own = job_duration(manifest)
        start = None
        finish = None
        if rate and unknown == 0:
            start = ahead / rate
            if own is not None:
                finish = (ahead + own) / rate
        jobs[manifest["id"]] = {
            "position": position,
            "seconds_ahead": round(ahead, 1),
            "unknown_ahead": unknown,
            "estimated_start_seconds": None if start is None else round(start, 1),
            "estimated_finish_seconds": None if finish is None else round(finish, 1),
            "estimate_is_minimum": bool(live_active),
        }
        if own is None:
            unknown += 1
            unknown_queued += 1
        else:
            ahead += own
            queued_seconds += own

    summary = {
        "queued": len(queued),
        "running": len(running),
        "queued_seconds": round(queued_seconds, 1),
        "unknown_duration": unknown_queued + sum(1 for m in running if _remaining(m) is None),
    }
    return {"summary": summary, "jobs": jobs}
