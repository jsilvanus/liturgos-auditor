"""Fetching a source file from a URL into a job's source directory.

Shared by the submit route (which fetches at submit time) and the job runner
(which fetches as the fallback when the fleet cannot strip the audio).
"""

import httpx


class UploadTooLargeError(Exception):
    pass


class FetchError(Exception):
    pass


def fetch_url(url, target, limit, timeout):
    """Stream `url` into `target`; no redirects are followed. Returns the size in bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with httpx.stream("GET", url, follow_redirects=False, timeout=timeout) as response:
            if response.status_code != 200:
                raise FetchError(f"source_url answered {response.status_code}")
            with open(target, "wb") as out:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise UploadTooLargeError()
                    out.write(chunk)
    except httpx.HTTPError:
        raise FetchError("source_url could not be fetched") from None
    return size
