#!/usr/bin/env python3
"""Transcribe a video with liturgos-auditor-stt and append WebVTT cues per chunk.

Requires:
  - Python 3.9+
  - ffmpeg and ffprobe on PATH
  - a running liturgos-auditor-stt service

Example:
  ./scripts/video-to-vtt.py sermon.mp4 sermon.vtt
  AUDITOR_STT_URL=http://localhost:8090 ./scripts/video-to-vtt.py sermon.mp4 sermon.vtt
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PUNCTUATION_NO_SPACE = frozenset(".,!?;:%)]}»”")
OPENING_PUNCTUATION = frozenset("([{«“")


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def join_words(words: list[dict[str, Any]]) -> str:
    text = ""
    for word in words:
        token = str(word.get("text", "")).strip()
        if not token:
            continue
        if not text:
            text = token
        elif token[0] in PUNCTUATION_NO_SPACE:
            text += token
        elif text[-1] in OPENING_PUNCTUATION:
            text += token
        else:
            text += " " + token
    return text


def split_lines(text: str, max_line_chars: int) -> str:
    words = text.split()
    if not words:
        return ""

    lines: list[str] = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if len(lines[-1]) < max_line_chars and len(candidate) <= max_line_chars:
            lines[-1] = candidate
        elif len(lines) == 1:
            lines.append(word)
        else:
            lines[-1] += " " + word

    return "\n".join(lines)


def make_cues(
    words: list[dict[str, Any]],
    max_duration: float,
    max_chars: int,
) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        start = float(current[0]["start"])
        end = float(current[-1]["end"])
        text = split_lines(join_words(current), max_chars)
        if text:
            cues.append((start, max(end, start), text))
        current.clear()

    for word in words:
        if "start" not in word or "end" not in word:
            continue

        if not current:
            current.append(word)
            continue

        candidate = current + [word]
        candidate_text = join_words(candidate)
        elapsed = float(word["end"]) - float(current[0]["start"])

        # Prefer punctuation as a natural cue boundary, while keeping cues
        # within the configured maximum duration.
        previous_text = join_words(current)
        punctuation_break = previous_text.endswith((".", "!", "?", ":", ";"))
        too_long = elapsed > max_duration
        too_wide = len(candidate_text.replace(" ", "")) > max_chars * 2

        if too_long or too_wide or punctuation_break:
            flush()
        current.append(word)

    flush()
    return cues


def get_duration(video: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not determine video duration: {result.stdout!r}") from exc
    if duration <= 0:
        raise RuntimeError(f"Video duration is invalid: {duration}")
    return duration


def extract_audio_chunk(
    video: Path,
    audio: Path,
    start: float,
    duration: float,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(audio),
    ]
    subprocess.run(command, check=True)


def transcribe(audio: Path, url: str, language: str) -> dict[str, Any]:
    boundary = "----liturgos-auditor-stt"
    audio_data = audio.read_bytes()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{audio.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio_data + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="language"\r\n\r\n'
        f"{language}\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    request = urllib.request.Request(
        url.rstrip("/") + "/inference",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"STT request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach STT service: {exc.reason}") from exc


def collect_words(result: dict[str, Any], offset: float) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        for word in segment.get("words") or []:
            if "start" not in word or "end" not in word:
                continue
            words.append(
                {
                    **word,
                    "start": float(word["start"]) + offset,
                    "end": float(word["end"]) + offset,
                }
            )
    return sorted(words, key=lambda word: float(word["start"]))


def write_vtt_header(output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("WEBVTT\n\n")


def append_cues(
    output: Path,
    cues: list[tuple[float, float, str]],
) -> int:
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for start, end, text in cues:
            handle.write(
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
                f"{html.escape(text)}\n\n"
            )
    return len(cues)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a video with liturgos-auditor-stt and append WebVTT cues per chunk."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--language",
        default="fi",
        help="STT language (default: fi)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("AUDITOR_STT_URL", "http://localhost:8090"),
        help="STT service base URL (default: AUDITOR_STT_URL or http://localhost:8090)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        help="Nominal chunk length in seconds (default: 60)",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=5.0,
        help="Overlap between consecutive chunks in seconds (default: 5)",
    )
    parser.add_argument(
        "--max-cue-duration",
        type=float,
        default=7.0,
        help="Maximum cue duration in seconds (default: 7)",
    )
    parser.add_argument(
        "--max-line-chars",
        type=int,
        default=42,
        help="Target maximum characters per caption line (default: 42)",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be greater than zero")
    if args.overlap_seconds < 0 or args.overlap_seconds >= args.chunk_seconds:
        parser.error("--overlap-seconds must be >= 0 and smaller than --chunk-seconds")
    if args.max_cue_duration <= 0:
        parser.error("--max-cue-duration must be greater than zero")
    if args.max_line_chars <= 0:
        parser.error("--max-line-chars must be greater than zero")

    output = args.output or args.video.with_suffix(".vtt")

    try:
        duration = get_duration(args.video)
        step = args.chunk_seconds - args.overlap_seconds
        write_vtt_header(output)

        total_cues = 0
        chunk_number = 0

        with tempfile.TemporaryDirectory(prefix="auditor-stt-") as temp_dir:
            audio = Path(temp_dir) / "chunk.wav"
            start = 0.0

            while start < duration:
                chunk_number += 1
                chunk_end = min(duration, start + args.chunk_seconds)
                chunk_duration = chunk_end - start

                print(
                    f"Chunk {chunk_number}: "
                    f"{format_timestamp(start)} -> {format_timestamp(chunk_end)} "
                    f"(STT window {chunk_duration:.1f}s)",
                    file=sys.stderr,
                )

                extract_audio_chunk(
                    args.video,
                    audio,
                    start,
                    chunk_duration,
                )
                result = transcribe(audio, args.url, args.language)
                words = collect_words(result, start)

                # Each word belongs to exactly one output interval. The
                # overlap gives the model context across chunk boundaries,
                # while the nominal step prevents duplicate VTT cues.
                output_start = start
                output_end = min(duration, start + step)
                if chunk_end >= duration:
                    output_end = duration

                owned_words = [
                    word
                    for word in words
                    if output_start <= float(word["start"]) < output_end
                ]
                cues = make_cues(
                    owned_words,
                    args.max_cue_duration,
                    args.max_line_chars,
                )
                total_cues += append_cues(output, cues)

                print(
                    f"  appended {len(cues)} cues; "
                    f"progress through {format_timestamp(output_end)}",
                    file=sys.stderr,
                )

                if chunk_end >= duration:
                    break
                start += step

        if total_cues == 0:
            raise RuntimeError("STT returned no word timestamps")

        print(f"Wrote {total_cues} cues to {output}", file=sys.stderr)
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
