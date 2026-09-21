#!/usr/bin/env python3
"""Transcribe a video with the liturgos-auditor-stt jobs API and write subtitles.

The service transcribes in ~60 s chunks and keeps every finished chunk on disk,
so an aborted run is not lost: the job id is printed as soon as the job is
accepted, and `--resume JOB_ID` picks the job up again (even after this script
was killed, or the service restarted).

Requires:
  - Python 3.9+
  - ffmpeg on PATH (unless --source-path or --no-extract)
  - a running liturgos-auditor-stt service with jobs enabled

Example:
  ./scripts/video-to-vtt-jobs.py sermon.mp4 sermon.vtt
  AUDITOR_STT_URL=http://localhost:8090 ./scripts/video-to-vtt-jobs.py sermon.mp4 sermon.vtt
  ./scripts/video-to-vtt-jobs.py --resume 3f2a... sermon.vtt
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

TRANSIENT_STATUSES = (502, 503, 504)
OUTPUT_SUFFIXES = {"vtt": ".vtt", "srt": ".srt", "youtube": ".txt", "json": ".json"}


class TransientHTTPError(RuntimeError):
    """A 502/503/504: the service or a proxy in front of it is briefly unavailable."""


def extract_audio(video: Path, audio: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio),
    ]
    subprocess.run(command, check=True)


# --- HTTP helpers -------------------------------------------------------------


def _connect(url: str) -> HTTPConnection:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    if parsed.scheme == "https":
        return HTTPSConnection(host, parsed.port or 443, timeout=30)
    return HTTPConnection(host, parsed.port or 80, timeout=30)


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _error_detail(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    try:
        detail = json.loads(text).get("detail", text)
    except (ValueError, AttributeError):
        return text
    return detail if isinstance(detail, str) else json.dumps(detail)


def _check_status(status: int, data: bytes, what: str, ok: tuple = (200,), retryable: bool = False) -> None:
    if status in ok:
        return
    error = TransientHTTPError if retryable and status in TRANSIENT_STATUSES else RuntimeError
    raise error(f"{what} failed ({status}): {_error_detail(data)}")


def _request(url: str, method: str, path: str, api_key: str | None, accept: str = "application/json") -> tuple:
    """One short request. Returns (status, headers, body)."""
    conn = _connect(url)
    try:
        conn.request(method, path, headers={"Accept": accept, **_auth_headers(api_key)})
        response = conn.getresponse()
        return response.status, response.headers, response.read()
    finally:
        conn.close()


def retry_with_backoff(fn, max_attempts: int = 10, base_delay: float = 0.5, retry_on: tuple | None = None):
    """Retry `fn` on transient failures with exponential backoff.

    By default that means connection-level errors and 502/503/504 answers: the job
    keeps running on the server while this client waits it out.
    """
    transient = retry_on or (
        BrokenPipeError,
        ConnectionRefusedError,
        ConnectionResetError,
        TimeoutError,
        socket.timeout,
        http.client.RemoteDisconnected,
        TransientHTTPError,
    )
    attempt = 0
    while True:
        try:
            return fn()
        except transient as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise RuntimeError(f"Request failed after {max_attempts} attempts: {exc}") from exc
            time.sleep(base_delay * (2 ** (attempt - 1)))


# --- multipart upload (streamed, so a large file is never held in memory) -----


def compute_multipart_length(
    file_size: int | None,
    boundary: str,
    language: str,
    chunk_seconds: str,
    source_path: str | None = None,
) -> int:
    """Compute the total Content-Length for the multipart body."""
    length = 0

    if source_path is None and file_size is not None:
        # file part header
        length += len(f"--{boundary}\r\n".encode())
        length += len(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
        length += len(b"Content-Type: audio/wav\r\n\r\n")
        length += file_size
        length += len(b"\r\n")
    elif source_path is not None:
        # source_path field
        length += len(f"--{boundary}\r\n".encode())
        length += len(b'Content-Disposition: form-data; name="source_path"\r\n\r\n')
        length += len(source_path.encode())
        length += len(b"\r\n")

    # language field
    length += len(f"--{boundary}\r\n".encode())
    length += len(b'Content-Disposition: form-data; name="language"\r\n\r\n')
    length += len(language.encode())
    length += len(b"\r\n")

    # chunk_seconds field
    length += len(f"--{boundary}\r\n".encode())
    length += len(b'Content-Disposition: form-data; name="chunk_seconds"\r\n\r\n')
    length += len(chunk_seconds.encode())
    length += len(b"\r\n")

    # closing boundary
    length += len(f"--{boundary}--\r\n".encode())

    return length


class MultipartStreamer:
    """File-like body: reads the multipart form (file or source_path, then fields) block by block."""

    def __init__(
        self,
        file_path: Path | None,
        boundary: str,
        language: str,
        chunk_seconds: str,
        source_path: str | None = None,
    ):
        self.file_path = file_path
        self.boundary = boundary
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.source_path = source_path
        self.file = None
        self.state = 0  # 0=initial, 1=file, 2=fields, 3=done

    def read(self, size: int = 8192) -> bytes:
        """Read up to size bytes of multipart body."""
        result = b""

        while len(result) < size and self.state < 3:
            if self.state == 0:
                if self.source_path is None:
                    header = (
                        f"--{self.boundary}\r\n"
                        'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                        "Content-Type: audio/wav\r\n\r\n"
                    ).encode()
                    result += header
                    self.state = 1
                else:
                    field = (
                        f"--{self.boundary}\r\n"
                        'Content-Disposition: form-data; name="source_path"\r\n\r\n'
                        f"{self.source_path}\r\n"
                    ).encode()
                    result += field
                    self.state = 2

            elif self.state == 1:
                if self.file is None:
                    self.file = self.file_path.open("rb")

                chunk = self.file.read(min(size - len(result), 1024 * 1024))
                if chunk:
                    result += chunk
                else:
                    self.file.close()
                    self.file = None
                    result += b"\r\n"
                    self.state = 2

            elif self.state == 2:
                fields = (
                    f"--{self.boundary}\r\n"
                    'Content-Disposition: form-data; name="language"\r\n\r\n'
                    f"{self.language}\r\n"
                    f"--{self.boundary}\r\n"
                    'Content-Disposition: form-data; name="chunk_seconds"\r\n\r\n'
                    f"{self.chunk_seconds}\r\n"
                    f"--{self.boundary}--\r\n"
                ).encode()
                result += fields
                self.state = 3

        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()


# --- the API ------------------------------------------------------------------


def check_service(url: str, api_key: str | None = None) -> None:
    """A cheap authenticated call, so a bad key or disabled jobs show up BEFORE a long upload.

    The service refuses unauthenticated and oversized uploads before reading the body and
    closes the connection, which a client only sees as a reset; this gives the real reason.
    """
    status, _, data = _request(url, "GET", f"/v1/jobs?client_ref={quote('auditor-stt-preflight')}", api_key)
    if status == 401:
        raise RuntimeError("The service rejected the API key (set it in the environment variable named by --api-key-env)")
    _check_status(status, data, "Service check", retryable=True)


def submit_job(
    url: str,
    file_path: Path | None,
    language: str,
    chunk_seconds: str,
    source_path: str | None = None,
    api_key: str | None = None,
) -> str:
    """Submit a transcription job. Returns the job id."""
    boundary = f"----auditor-stt-{uuid.uuid4().hex}"
    file_size = file_path.stat().st_size if file_path else None
    content_length = compute_multipart_length(file_size, boundary, language, chunk_seconds, source_path)

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(content_length),
        "Accept": "application/json",
        **_auth_headers(api_key),
    }

    conn = _connect(url)
    try:
        with MultipartStreamer(file_path, boundary, language, chunk_seconds, source_path) as streamer:
            try:
                conn.request("POST", "/v1/jobs", streamer, headers=headers)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
                # The service answers a refused upload and then closes the socket; the answer may still be readable.
                try:
                    response = conn.getresponse()
                    data = response.read()
                except Exception:  # noqa: BLE001 - nothing readable: fall through to the generic message
                    raise RuntimeError(
                        "The service closed the connection during the upload "
                        "(wrong or missing API key, or the file exceeds its upload limit?)"
                    ) from exc
                _check_status(response.status, data, "Job submission", ok=(200, 202))
                raise RuntimeError("The service closed the connection during the upload") from exc

            response = conn.getresponse()
            data = response.read()
            _check_status(response.status, data, "Job submission", ok=(200, 202))
            return json.loads(data)["id"]
    finally:
        conn.close()


def get_job_status(url: str, job_id: str, api_key: str | None = None) -> dict[str, Any]:
    status, _, data = _request(url, "GET", f"/v1/jobs/{job_id}", api_key)
    _check_status(status, data, "Job status request", retryable=True)
    return json.loads(data)


def fetch_job_result(
    url: str,
    job_id: str,
    format: str = "vtt",
    partial: bool = False,
    max_cue_duration: float | None = None,
    max_line_chars: int | None = None,
    start_time: str | None = None,
    api_key: str | None = None,
) -> tuple[str, bool]:
    """Fetch a job's result document. Returns (content, is_complete)."""
    params = [f"format={format}", f"partial={'true' if partial else 'false'}"]
    if max_cue_duration is not None:
        params.append(f"max_cue_duration={max_cue_duration}")
    if max_line_chars is not None:
        params.append(f"max_line_chars={max_line_chars}")
    if start_time:
        params.append(f"start_time={quote(start_time)}")

    accept = "application/json" if format == "json" else "text/plain"
    status, headers, data = _request(url, "GET", f"/v1/jobs/{job_id}/result?" + "&".join(params), api_key, accept)
    _check_status(status, data, "Result fetch", retryable=True)
    return data.decode("utf-8"), headers.get("X-Job-Complete", "true").lower() == "true"


def delete_job(url: str, job_id: str, api_key: str | None = None) -> None:
    status, _, data = _request(url, "DELETE", f"/v1/jobs/{job_id}", api_key)
    _check_status(status, data, "Job deletion", ok=(202, 204))


def write_atomic(path: Path, content: str) -> None:
    """Write file atomically using a temporary sibling."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # Bytes, not write_text: text mode would turn "\n" into "\r\n" on Windows, and the service's
    # youtube format is "\n"-separated on the wire; the file should hold exactly what it sent.
    tmp_path.write_bytes(content.encode("utf-8"))
    tmp_path.replace(path)


def format_progress(status: dict[str, Any]) -> str:
    msg = f"{float(status.get('progress') or 0):.1f}%"
    if status.get("chunks_total"):
        msg += f" ({status.get('chunks_done', 0)}/{status['chunks_total']} chunks)"
    if status.get("eta_seconds") is not None:
        msg += f" ETA {int(status['eta_seconds'])}s"
    return msg


# --- command line -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a video with liturgos-auditor-stt and write subtitle files."
    )
    parser.add_argument(
        "video", type=Path, nargs="?", help="Video file (not needed with --source-path or --resume)"
    )
    parser.add_argument(
        "output", type=Path, nargs="?", help="Output file (default: VIDEO.FORMAT or derived from --source-path)"
    )
    parser.add_argument("--language", default="fi", help="STT language (default: fi)")
    parser.add_argument(
        "--url",
        default=os.environ.get("AUDITOR_STT_URL", "http://localhost:8090"),
        help="STT service base URL (default: AUDITOR_STT_URL or http://localhost:8090)",
    )
    parser.add_argument(
        "--api-key-env",
        default="AUDITOR_STT_API_KEY",
        help="Environment variable holding the API key, if the service requires one (default: AUDITOR_STT_API_KEY)",
    )
    parser.add_argument("--chunk-seconds", default="60", help="Chunk size in seconds (default: 60)")
    parser.add_argument(
        "--format",
        choices=list(OUTPUT_SUFFIXES),
        default="vtt",
        help="Output format (default: vtt)",
    )
    parser.add_argument("--start-time", help="Wall-clock time of the video's start, ISO UTC (required for youtube)")
    parser.add_argument("--max-cue-duration", type=float, default=7, help="Maximum cue duration in seconds (default: 7)")
    parser.add_argument(
        "--max-line-chars",
        type=int,
        default=42,
        help="Target maximum characters per caption line (default: 42)",
    )
    parser.add_argument("--poll-interval", type=float, default=3, help="Polling interval in seconds (default: 3)")
    parser.add_argument("--source-path", help="Submit a path readable by the SERVICE instead of uploading")
    parser.add_argument("--no-extract", action="store_true", help="Upload the file as is instead of extracting audio")
    parser.add_argument("--resume", metavar="JOB_ID", help="Skip submission: poll this job and fetch its result")
    parser.add_argument("--keep-job", action="store_true", help="Do not delete the job on the server after success")
    parser.add_argument("--json-out", type=Path, help="Also save the JSON transcript")
    parser.add_argument(
        "--cancel-on-abort",
        action="store_true",
        help="Cancel the job on Ctrl-C instead of leaving it running",
    )

    args = parser.parse_args()

    # There is no video with --source-path or --resume, so a lone positional is the OUTPUT, as in the
    # usage examples; argparse would otherwise hand it to VIDEO and drop it (or demand an OUTPUT).
    if args.output is None and args.video is not None and (args.source_path or args.resume):
        args.output, args.video = args.video, None

    if args.format == "youtube" and not args.start_time:
        parser.error("--start-time is required for youtube format")

    if args.resume:
        job_id = args.resume
        output = args.output
        if not output:
            parser.error("OUTPUT is required with --resume")
    else:
        if not args.source_path and args.video is None:
            parser.error("VIDEO is required unless using --source-path or --resume")
        if not args.source_path and not args.video.is_file():
            parser.error(f"Video does not exist: {args.video}")

        output = args.output
        if not output:
            base = Path(args.source_path).stem if args.source_path else args.video.stem
            output = Path(base + OUTPUT_SUFFIXES[args.format])

    api_key = os.environ.get(args.api_key_env)
    job_id_global = None

    def handle_interrupt(sig, frame):
        if job_id_global and args.cancel_on_abort:
            try:
                print(f"Cancelling job {job_id_global}...", file=sys.stderr)
                delete_job(args.url, job_id_global, api_key)
            except Exception as exc:  # noqa: BLE001 - exiting anyway
                print(f"Failed to cancel job: {exc}", file=sys.stderr)
        elif job_id_global:
            print(
                f"Interrupted. The job keeps running on the server; resume with: --resume {job_id_global}",
                file=sys.stderr,
            )
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        if not args.resume:
            # Before any ffmpeg work or upload: a wrong key or disabled jobs should fail fast, with the real reason.
            retry_with_backoff(
                lambda: check_service(args.url, api_key),
                max_attempts=3,
                retry_on=(ConnectionRefusedError, TransientHTTPError),
            )

            def submit(upload: Path | None) -> str:
                # Only a refused connection is retried: after that nothing was sent, so no duplicate job can result.
                return retry_with_backoff(
                    lambda: submit_job(
                        args.url,
                        upload,
                        args.language,
                        args.chunk_seconds,
                        source_path=args.source_path,
                        api_key=api_key,
                    ),
                    retry_on=(ConnectionRefusedError,),
                )

            if args.source_path:
                print(f"Submitting job to {args.url} (source_path)", file=sys.stderr)
                job_id = submit(None)
            elif args.no_extract:
                print(f"Submitting job to {args.url}", file=sys.stderr)
                job_id = submit(args.video)
            else:
                with tempfile.TemporaryDirectory(prefix="auditor-stt-") as temp_dir:
                    audio_file = Path(temp_dir) / "audio.wav"
                    print(f"Extracting audio: {args.video}", file=sys.stderr)
                    extract_audio(args.video, audio_file)
                    print(f"Submitting job to {args.url}", file=sys.stderr)
                    job_id = submit(audio_file)

            print(f"Job {job_id} submitted (resume with: --resume {job_id})", file=sys.stderr)

        job_id_global = job_id

        is_tty = sys.stderr.isatty()
        last_progress = None

        while True:
            status = retry_with_backoff(lambda: get_job_status(args.url, job_id, api_key))
            state = status["status"]

            msg = format_progress(status)
            if last_progress != msg:
                last_progress = msg
                print(f"\r{msg}" if is_tty else msg, file=sys.stderr, end="" if is_tty else "\n", flush=True)

            if state == "completed":
                if is_tty:
                    print(file=sys.stderr)

                print("Fetching result", file=sys.stderr)
                content, _ = retry_with_backoff(
                    lambda: fetch_job_result(
                        args.url,
                        job_id,
                        args.format,
                        partial=False,
                        max_cue_duration=args.max_cue_duration,
                        max_line_chars=args.max_line_chars,
                        start_time=args.start_time,
                        api_key=api_key,
                    )
                )
                write_atomic(output, content)
                print(f"Wrote {output}", file=sys.stderr)

                if args.json_out:
                    json_content, _ = retry_with_backoff(
                        lambda: fetch_job_result(args.url, job_id, "json", partial=False, api_key=api_key)
                    )
                    write_atomic(args.json_out, json_content)
                    print(f"Wrote {args.json_out}", file=sys.stderr)

                # The transcript now lives in the output file; do not leave it on the server.
                if not args.keep_job:
                    try:
                        delete_job(args.url, job_id, api_key)
                    except Exception as exc:  # noqa: BLE001 - the result is already saved
                        print(f"Warning: could not delete the job on the server: {exc}", file=sys.stderr)

                return 0

            if state in ("failed", "cancelled"):
                if is_tty:
                    print(file=sys.stderr)

                # Each finished chunk is kept, so most of the file is usually still recoverable.
                print(f"Job {state}. Fetching partial result...", file=sys.stderr)
                try:
                    content, _ = retry_with_backoff(
                        lambda: fetch_job_result(
                            args.url,
                            job_id,
                            args.format,
                            partial=True,
                            max_cue_duration=args.max_cue_duration,
                            max_line_chars=args.max_line_chars,
                            start_time=args.start_time,
                            api_key=api_key,
                        )
                    )
                    partial_output = output.with_suffix(output.suffix + ".partial")
                    write_atomic(partial_output, content)
                    print(f"Wrote partial result to {partial_output}", file=sys.stderr)
                except Exception as exc:  # noqa: BLE001 - report it, the job failure is the main news
                    print(f"Failed to fetch partial result: {exc}", file=sys.stderr)

                if status.get("error"):
                    print(f"Error: {status['error']}", file=sys.stderr)
                print(
                    f"Completed {status.get('chunks_done', 0)}/{status.get('chunks_total', 0)} chunks; "
                    "the job record stays on the server until it expires",
                    file=sys.stderr,
                )
                return 3

            time.sleep(args.poll_interval)

    except Exception as exc:  # noqa: BLE001 - one message and exit code for anything unexpected
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
