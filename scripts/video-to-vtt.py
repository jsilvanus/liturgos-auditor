#!/usr/bin/env python3
"""Transcribe a video with liturgos-auditor-stt and write a WebVTT file.

Requires:
  - Python 3.9+
  - ffmpeg on PATH
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


def collect_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        words.extend(segment.get("words") or [])
    return sorted(words, key=lambda word: float(word.get("start", 0)))


def write_vtt(output: Path, cues: list[tuple[float, float, str]]) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("WEBVTT\n\n")
        for start, end, text in cues:
            handle.write(
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
                f"{html.escape(text)}\n\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a video with liturgos-auditor-stt and write WebVTT."
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

    output = args.output or args.video.with_suffix(".vtt")

    try:
        with tempfile.TemporaryDirectory(prefix="auditor-stt-") as temp_dir:
            audio = Path(temp_dir) / "audio.wav"
            print(f"Extracting audio: {args.video}", file=sys.stderr)
            extract_audio(args.video, audio)

            print(f"Transcribing with {args.url}", file=sys.stderr)
            result = transcribe(audio, args.url, args.language)

        words = collect_words(result)
        if not words:
            raise RuntimeError("STT returned no word timestamps")

        cues = make_cues(words, args.max_cue_duration, args.max_line_chars)
        write_vtt(output, cues)
        print(f"Wrote {len(cues)} cues to {output}", file=sys.stderr)
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
