"""`auditor-stt dataset sync` — keeps the local ledger in step with a crowd-source-voice corpus.

`dataset pull` re-downloads everything into a fresh snapshot. Sync instead
diffs the validated export against the ledger: only new recordings are
downloaded, changed text is updated in place, and a recording that disappears
upstream is tombstoned and its audio erased (csv hard-deletes without
notifying anyone, so absence from the listing is the only deletion signal).

Run one sync at a time per data directory.

Privacy: logs carry recording ids and counts only, never transcripts or audio.
Speaker ids are stored only in pseudonymous form; a raw csv user id is hashed
with a local salt or dropped, never persisted.
"""

import hashlib
import hmac
import logging
import os
import re
from dataclasses import dataclass, field

import httpx

from .audio import AudioNormalizeError, audio_path, normalize_audio, store_audio, wav_duration
from .client import CrowdSourceVoiceClient
from .ledger import Ledger, ledger_path, text_hash, utc_iso
from .normalize import normalize_text
from .pull import (
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    ExportInconsistentError,
    UnsupportedCorpusTypeError,
    _paired,
)

logger = logging.getLogger(__name__)

# A sync that would tombstone more than this share of a corpus (and more than
# this many rows) looks like a broken export, not real deletions.
MASS_REMOVAL_FRACTION = 0.5
MASS_REMOVAL_MIN_ROWS = 5

# csv user ids are sequential integers. The bound keeps a 24-char hex pseudonym that
# happens to be all digits (about 1 in 77,000) from being mistaken for one.
_RAW_USER_ID = re.compile(r"[0-9]{1,18}")


class MassRemovalError(Exception):
    """Refused to tombstone a suspiciously large part of a corpus."""


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: dict = field(default_factory=dict)

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def summary(self):
        text = f"added {self.added}, updated {self.updated}, removed {self.removed}, unchanged {self.unchanged}"
        if self.skipped:
            detail = ", ".join(f"{reason}={count}" for reason, count in sorted(self.skipped.items()))
            text += f"; skipped {sum(self.skipped.values())} ({detail})"
        return text

    __str__ = summary


@dataclass(frozen=True)
class Listed:
    """One recording in the upstream listing, whichever export version produced it."""

    recording_id: int
    audio_path: str | None
    text: str | None
    quality_score: float | None
    speaker_id: object


def hash_speaker_id(raw_id, salt):
    """Pseudonymise a raw csv user id: hex HMAC-SHA256(salt, id), truncated to 24 chars like csv's own."""
    return hmac.new(salt.encode("utf-8"), str(raw_id).encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def _resolve_speaker(value, salt, report):
    """Map the export's speaker_id to what may be stored: a pseudonym or None.

    A JSON int or all-digit string is a raw csv user id (hashed if there is a
    salt, else dropped). Anything email-like, or not a string or int, is
    dropped. Any other string is taken to be csv's pseudonymous id already.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        report.skip("unusable_speaker_id_dropped")
        return None
    if isinstance(value, int) or _RAW_USER_ID.fullmatch(value.strip()):
        if salt:
            return hash_speaker_id(str(value).strip(), salt)
        report.skip("raw_speaker_id_dropped")
        return None
    value = value.strip()
    if "@" in value:
        report.skip("unusable_speaker_id_dropped")
        return None
    return value or None


def _clean_path(path):
    return path.lstrip("/") if path else None


def _check_corpus(export, corpus_id):
    corpus = export["corpus"]
    if corpus is not None and corpus["type"] != "text":
        raise UnsupportedCorpusTypeError(
            f"Corpus {corpus_id} ('{corpus['name']}') is type '{corpus['type']}': "
            "auditor-stt only trains on 'text' corpora"
        )


def _entries_from_export(rows):
    return [
        Listed(
            recording_id=int(row["recording_id"]),
            audio_path=_clean_path(row.get("audio_url") or row.get("original_path")),
            text=normalize_text(row.get("text")),
            quality_score=row.get("quality_score"),
            speaker_id=row.get("speaker_id"),
        )
        for row in rows
    ]


def _entries_from_manifest(export, manifest):
    by_name = {entry["export_name"]: entry for entry in manifest["files"]}
    entries = []
    for row in export["recordings"]:
        entry = by_name[row["file"]]  # presence guaranteed by _paired()
        speaker_id = row.get("speaker_id")
        entries.append(
            Listed(
                recording_id=int(entry["id"]),
                audio_path=_clean_path(entry.get("source_path")),
                text=normalize_text(row.get("text")),
                quality_score=row.get("quality_score"),
                speaker_id=speaker_id if speaker_id is not None else entry.get("speaker_id"),
            )
        )
    return entries


def _fetch_listing(client, corpus_id):
    """The corpus's current validated recordings as `Listed` entries; empty for an empty corpus.

    Newer csv exports carry a stable `recording_id` per row and need only one
    call. Older ones name files by position, so rows are paired with the
    manifest by that name and verified (see `_paired`); the pairing is
    refetched once, and refused if it still disagrees.
    """
    for attempt in (1, 2):
        export = client.get_export_or_empty(corpus_id)
        _check_corpus(export, corpus_id)
        rows = export["recordings"]
        if not rows:
            return []
        if all(row.get("recording_id") is not None for row in rows):
            return _entries_from_export(rows)
        manifest = client.get_manifest_or_empty(corpus_id)
        if _paired(export, manifest):
            return _entries_from_manifest(export, manifest)
        if attempt == 1:
            logger.warning("Export and manifest disagree (data changed between calls?); refetching once")
    raise ExportInconsistentError(
        f"Export and manifest for corpus {corpus_id} still disagree after a refetch; "
        "refusing to pair text with audio by position. Try again when no validation is in progress."
    )


def _delete_audio(data_dir, sha):
    try:
        audio_path(data_dir, sha).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete audio file %s: %s", sha, type(exc).__name__)


def sync_corpus(
    client, ledger, corpus_id, data_dir, *, normalizer=None, speaker_salt=None, allow_mass_removal=False, now=None,
):
    """Bring the ledger's rows for `corpus_id` in line with the corpus's current export.

    `normalizer(bytes, source_name) -> wav_bytes` defaults to ffmpeg to 16 kHz
    mono PCM16 (tests inject a stub). Raises UnsupportedCorpusTypeError for
    non-'text' corpora, ExportInconsistentError when a legacy export cannot be
    paired reliably, and MassRemovalError (before changing anything) when the
    listing is missing most of the ledger's rows and `allow_mass_removal` is
    not set. One recording that fails to download or convert is counted in
    `skipped` and never aborts the run or tombstones anything.
    """
    normalizer = normalizer or normalize_audio
    now = utc_iso(now)
    report = SyncReport()

    entries = _fetch_listing(client, corpus_id)
    listed_ids = {entry.recording_id for entry in entries}

    active = {row["recording_id"]: row for row in ledger.active_rows(corpus_id)}
    gone = [rid for rid in active if rid not in listed_ids]
    if (
        gone
        and not allow_mass_removal
        and len(gone) > MASS_REMOVAL_MIN_ROWS
        and len(gone) > len(active) * MASS_REMOVAL_FRACTION
    ):
        raise MassRemovalError(
            f"Refusing to remove {len(gone)} of {len(active)} recordings from corpus {corpus_id}: "
            "the export no longer lists them. An empty or partial export usually means a broken export or "
            "the wrong corpus. Pass --allow-mass-removal if these deletions are genuine."
        )

    # Erase removals first so a later failure cannot delay them. The file goes
    # before the tombstone: if we stop in between, the next run still sees the
    # id missing and finishes the job, whereas the reverse order could orphan
    # a voice recording that nothing references.
    for sha in {active[rid]["audio_sha256"] for rid in gone} - {None}:
        if not ledger.audio_hash_in_use(sha, exclude_ids=gone):
            _delete_audio(data_dir, sha)
    report.removed = ledger.mark_removed(gone, now)

    ledger.mark_seen(listed_ids, now)

    for entry in entries:
        rid = entry.recording_id
        # Not in `active` means new, a tombstone being revived, or a row filed under another corpus.
        known = active.get(rid) or ledger.get(rid)
        speaker_id = _resolve_speaker(entry.speaker_id, speaker_salt, report)
        if not entry.text:
            report.skip("empty_text")
            continue

        is_active = known is not None and known["status"] == "active"
        audio_current = (
            is_active
            and known["audio_sha256"] is not None
            and known["audio_path"] == entry.audio_path
            and audio_path(data_dir, known["audio_sha256"]).exists()
        )
        if audio_current:
            same = (known["corpus_id"], known["text_hash"], known["quality_score"], known["speaker_id"]) == (
                corpus_id, text_hash(entry.text), entry.quality_score, speaker_id,
            )
            if same:
                report.unchanged += 1
            else:
                ledger.upsert_active(
                    rid, corpus_id, speaker_id=speaker_id, text=entry.text, audio_sha256=known["audio_sha256"],
                    duration=known["duration"], quality_score=entry.quality_score, audio_path=known["audio_path"],
                    now=now,
                )
                report.updated += 1
            continue

        if not entry.audio_path:
            report.skip("missing_audio_path")
            continue
        try:
            raw = client.download_audio(entry.audio_path)
        except httpx.HTTPError as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            logger.warning("Download failed for recording %s (%s %s)", rid, type(exc).__name__, status or "")
            report.skip("download_failed")
            continue
        try:
            wav = normalizer(raw, entry.audio_path)
            duration = wav_duration(wav)
        except AudioNormalizeError as exc:
            logger.warning("Could not convert recording %s: %s", rid, exc)
            report.skip("audio_undecodable")
            continue
        # csv's own duration field is client-supplied and never checked server-side; trust only the audio.
        if not (MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS):
            logger.warning("Skipping recording %s: duration %.2fs outside [%s, %s]", rid, duration, MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
            report.skip("duration_out_of_range")
            continue

        sha = hashlib.sha256(wav).hexdigest()
        store_audio(data_dir, sha, wav)
        ledger.upsert_active(
            rid, corpus_id, speaker_id=speaker_id, text=entry.text, audio_sha256=sha, duration=duration,
            quality_score=entry.quality_score, audio_path=entry.audio_path, now=now,
        )
        if is_active:
            report.updated += 1
            old_sha = known["audio_sha256"]
            if old_sha and old_sha != sha and not ledger.audio_hash_in_use(old_sha):
                _delete_audio(data_dir, old_sha)
        else:
            report.added += 1

    if report.skipped.get("raw_speaker_id_dropped"):
        logger.warning(
            "%d recordings carry a raw csv user id and no speaker salt is set; they were stored without a speaker id",
            report.skipped["raw_speaker_id_dropped"],
        )
    logger.info("Synced corpus %s: %s", corpus_id, report)
    return report


def run_sync(
    base_url, corpus_id, data_dir, token_env="CSV_ADMIN_TOKEN", speaker_salt_env="AUDITOR_STT_SPEAKER_SALT",
    allow_mass_removal=False, client=None, normalizer=None,
):
    """Open the ledger under `data_dir` and sync one corpus, reading the token and salt from env vars."""
    owns_client = client is None
    if client is None:
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f"Env var {token_env} is not set; a crowd-source-voice bearer token is required")
        client = CrowdSourceVoiceClient(base_url, token)
    salt = os.environ.get(speaker_salt_env) or None

    try:
        with Ledger(ledger_path(data_dir)) as ledger:
            return sync_corpus(
                client, ledger, corpus_id, data_dir,
                normalizer=normalizer, speaker_salt=salt, allow_mass_removal=allow_mass_removal,
            )
    finally:
        if owns_client:
            client.close()
