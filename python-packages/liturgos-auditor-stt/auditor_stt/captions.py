"""Caption formats built from a whisper-style transcript result.

Cue grouping is ported from scripts/video-to-vtt.py, which stays a standalone
stdlib script, so the service and the script cut cues the same way. The input
is the JSON result shape (`segments[].words[]` with start/end/text), with
absolute timestamps.
"""

import html
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

PUNCTUATION_NO_SPACE = frozenset(".,!?;:%)]}»”")
OPENING_PUNCTUATION = frozenset("([{«“")

# lcyt's sender defaults; used when only one of region/cue is given.
DEFAULT_REGION = "reg1"
DEFAULT_CUE = "cue1"


@dataclass
class Cue:
    start: float
    end: float
    text: str  # may contain "\n" between wrapped lines


def format_timestamp(seconds, decimal="."):
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal}{milliseconds:03d}"


def join_words(words):
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


def split_lines(text, max_line_chars):
    words = text.split()
    if not words:
        return ""

    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if len(lines[-1]) < max_line_chars and len(candidate) <= max_line_chars:
            lines[-1] = candidate
        elif len(lines) == 1:
            lines.append(word)
        else:
            lines[-1] += " " + word

    return "\n".join(lines)


def make_cues(result, max_duration=7.0, max_chars=42):
    """Group a result's words into cues; a segment without word timings becomes one cue.

    Words keep grouping across consecutive segments (as the script does); a
    word-less segment ends the running cue so cues stay in time order.
    """
    cues = []
    current = []

    def flush():
        if not current:
            return
        start = float(current[0]["start"])
        end = float(current[-1]["end"])
        text = split_lines(join_words(current), max_chars)
        if text:
            cues.append(Cue(start, max(end, start), text))
        current.clear()

    for segment in result.get("segments") or []:
        words = [w for w in (segment.get("words") or []) if "start" in w and "end" in w]

        if not words:
            flush()
            text = split_lines(str(segment.get("text", "")), max_chars)
            if text:
                start = float(segment["start"])
                cues.append(Cue(start, max(float(segment.get("end", start)), start), text))
            continue

        for word in words:
            if current:
                elapsed = float(word["end"]) - float(current[0]["start"])
                candidate_text = join_words(current + [word])

                # Prefer punctuation as a natural cue boundary, while keeping cues
                # within the configured maximum duration.
                punctuation_break = join_words(current).endswith((".", "!", "?", ":", ";"))
                too_long = elapsed > max_duration
                too_wide = len(candidate_text.replace(" ", "")) > max_chars * 2

                if too_long or too_wide or punctuation_break:
                    flush()
            current.append(word)

    flush()
    return cues


def to_vtt(cues):
    parts = ["WEBVTT\n\n"]
    for cue in cues:
        # WebVTT only defines &amp; &lt; &gt; references, so quotes stay literal.
        parts.append(
            f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n"
            f"{html.escape(cue.text, quote=False)}\n\n"
        )
    return "".join(parts)


def to_srt(cues):
    parts = []
    for index, cue in enumerate(cues, start=1):
        parts.append(
            f"{index}\n"
            f"{format_timestamp(cue.start, ',')} --> {format_timestamp(cue.end, ',')}\n"
            f"{cue.text}\n\n"
        )
    return "".join(parts)


def to_text(result):
    text = result.get("text")
    if text is None:
        text = " ".join(str(s.get("text", "")).strip() for s in result.get("segments") or [])
    return " ".join(text.split())


def parse_start_time(value):
    """Parse an ISO-8601 instant into an aware datetime; a naive value is taken as UTC.

    datetime.fromisoformat rejects a trailing "Z" before Python 3.11.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def to_youtube(cues, start_time, region=None, cue=None):
    """YouTube Live Captions ingestion body, byte-compatible with lcyt's sender.

    One record per cue: `YYYY-MM-DDTHH:MM:SS.mmm` (absolute UTC, no offset,
    milliseconds truncated because YouTube rejects microseconds), optionally
    ` region:<region>#<cue>`, a newline, and the text on ONE line. Records are
    joined by "\\n" with a trailing "\\n". `start_time` is the wall-clock time
    of media t=0 (naive is taken as UTC). Empty input gives "".
    """
    if start_time.tzinfo is None:
        start_utc = start_time.replace(tzinfo=timezone.utc)
    else:
        start_utc = start_time.astimezone(timezone.utc)

    suffix = ""
    if region is not None or cue is not None:
        region = region or DEFAULT_REGION
        cue = cue or DEFAULT_CUE
        # Whitespace would break the record structure the same way a newline in the text would.
        if any(ch.isspace() for ch in region + cue):
            raise ValueError("region and cue must not contain whitespace")
        suffix = f" region:{region}#{cue}"

    records = []
    for item in cues:
        text = " ".join(item.text.split())
        if not text:
            continue
        moment = start_utc + timedelta(milliseconds=max(0, round(item.start * 1000)))
        stamp = f"{moment.strftime('%Y-%m-%dT%H:%M:%S')}.{moment.microsecond // 1000:03d}"
        records.append(f"{stamp}{suffix}\n{text}")

    return "\n".join(records) + "\n" if records else ""
