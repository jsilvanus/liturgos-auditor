"""`auditor-stt dataset pull` — turns a crowd-source-voice export into an immutable training snapshot."""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from .client import CrowdSourceVoiceClient
from .normalize import normalize_text

logger = logging.getLogger(__name__)

# Client-side capture gating (audioRecorder.js) enforces this at record time,
# but crowd-source-voice's server never re-validates it on upload — re-check
# here rather than trusting export metadata blindly.
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 30


class UnsupportedCorpusTypeError(Exception):
    pass


class ExportInconsistentError(Exception):
    """The export and manifest responses do not describe the same recordings."""


def _paired(export, manifest):
    """True when export and manifest agree row for row.

    crowd-source-voice names files by position within each response
    (`0001.wav`, ...), so a recording crossing the validation threshold between
    the two calls shifts every later name and would pair one recording's text
    with another's audio. The manifest repeats each row's text, so compare it.
    """
    rows, files = export["recordings"], manifest["files"]
    if len(rows) != len(files):
        return False
    by_name = {entry["export_name"]: entry for entry in files}
    for row in rows:
        entry = by_name.get(row["file"])
        if entry is None:
            return False
        if "text" in entry and normalize_text(entry["text"]) != normalize_text(row.get("text")):
            return False
    return True


def pull_snapshot(base_url, corpus_id, out_dir, token_env="CSV_ADMIN_TOKEN", client=None):
    """Pulls a validated snapshot for a 'text' corpus. Raises UnsupportedCorpusTypeError
    for 'music' corpora — those export a `notation` field instead of `text` and are out
    of scope for auditor-stt training."""
    owns_client = client is None
    if client is None:
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f"Env var {token_env} is not set — need an admin-role crowd-source-voice bearer token")
        client = CrowdSourceVoiceClient(base_url, token)

    try:
        export = client.get_export(corpus_id)

        corpus = export["corpus"]
        if corpus["type"] != "text":
            raise UnsupportedCorpusTypeError(
                f"Corpus {corpus_id} ('{corpus['name']}') is type '{corpus['type']}' — "
                "auditor-stt only trains on 'text' corpora"
            )

        manifest = client.get_manifest(corpus_id)

        if not _paired(export, manifest):
            logger.warning("Export and manifest disagree (data changed between calls?); refetching once")
            export = client.get_export(corpus_id)
            corpus = export["corpus"]
            manifest = client.get_manifest(corpus_id)
            if not _paired(export, manifest):
                raise ExportInconsistentError(
                    f"Export and manifest for corpus {corpus_id} still disagree after a refetch; "
                    "refusing to pair text with audio by position. Try again when no validation is in progress."
                )

        out = Path(out_dir)
        audio_dir = out / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        manifest_by_name = {row["export_name"]: row for row in manifest["files"]}

        rows = []
        total_duration = 0.0
        hasher = hashlib.sha256()

        for row in export["recordings"]:
            duration = row.get("duration") or 0
            if not (MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS):
                logger.warning("Skipping %s: duration %.2fs outside [%s, %s]", row["file"], duration, MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
                continue

            manifest_row = manifest_by_name[row["file"]]  # presence guaranteed by _paired()

            audio_bytes = client.download_audio(manifest_row["source_path"])
            (audio_dir / row["file"]).write_bytes(audio_bytes)
            hasher.update(audio_bytes)

            rows.append({
                "file": row["file"],
                "text": row["text"],
                "duration": duration,
                "quality_score": row.get("quality_score"),
                # None on corpora exported before crowd-source-voice added speaker_id
                "speaker_id": row.get("speaker_id"),
            })
            total_duration += duration

        (out / "recordings.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        snapshot = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_id": corpus_id,
            "corpus_name": corpus["name"],
            "language": corpus["language"],
            "recording_count": len(rows),
            "total_duration_seconds": total_duration,
            "content_hash": hasher.hexdigest(),
            "has_speaker_id": any(r["speaker_id"] for r in rows),
        }
        (out / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Pulled snapshot: %s", snapshot)
        return snapshot
    finally:
        if owns_client:
            client.close()
