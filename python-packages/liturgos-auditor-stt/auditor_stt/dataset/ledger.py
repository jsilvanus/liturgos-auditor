"""SQLite ledger of the crowd-source-voice recordings this installation has synced.

The ledger, not a snapshot directory, is the source of truth for training data:
`dataset sync` keeps it in step with the upstream export, and dataset builds
read only its active rows. csv hard-deletes recordings without telling anyone,
so a recording that vanishes upstream is tombstoned here and everything
personal about it (text, audio hash, speaker) is erased on the spot. The
tombstone itself stays, so recording ids are never reused and lineage records
that mention a removed recording still resolve.

Only the pseudonymous speaker id is ever stored, never a raw csv user id.
"""

import hashlib
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
LEDGER_FILENAME = "ledger.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recordings (
    recording_id  INTEGER PRIMARY KEY,
    corpus_id     INTEGER NOT NULL,
    speaker_id    TEXT,
    text          TEXT,
    text_hash     TEXT,
    audio_sha256  TEXT,
    audio_path    TEXT,
    duration      REAL,
    quality_score REAL,
    validated_at  TEXT,
    status        TEXT NOT NULL CHECK (status IN ('active', 'removed')),
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    removed_at    TEXT
);
CREATE INDEX IF NOT EXISTS recordings_corpus_status ON recordings (corpus_id, status);
CREATE INDEX IF NOT EXISTS recordings_audio_sha256 ON recordings (audio_sha256);
CREATE INDEX IF NOT EXISTS recordings_speaker_id ON recordings (speaker_id);
"""

# SQLite caps bound parameters per statement (999 on older builds).
_BATCH = 500


class LedgerError(RuntimeError):
    pass


def ledger_path(data_dir):
    return Path(data_dir) / LEDGER_FILENAME


def utc_iso(value=None):
    """ISO-8601 UTC string (second precision, `Z` suffix) from a datetime, an
    already-formatted string, or now. Naive datetimes are taken to be UTC."""
    if isinstance(value, str):
        return value
    if value is None:
        value = datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def text_hash(text):
    """sha256 of the NFC-normalised UTF-8 text; the stable identity of a transcript."""
    if text is None:
        return None
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


def _batches(items):
    items = list(items)
    for start in range(0, len(items), _BATCH):
        yield items[start:start + _BATCH]


class Ledger:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Zero freed pages so a tombstoned row's text does not linger in the file.
        self._conn.execute("PRAGMA secure_delete=ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
        found = int(self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0])
        if found != SCHEMA_VERSION:
            self._conn.close()
            raise LedgerError(f"Ledger {path} has schema version {found}; this build understands {SCHEMA_VERSION}")

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def upsert_active(
        self, recording_id, corpus_id, *, speaker_id, text, audio_sha256, duration, quality_score,
        audio_path=None, validated_at=None, now=None,
    ):
        """Insert a recording, or refresh a known one (also reviving a tombstone).

        `first_seen` is kept across updates and revivals. `text` is stored as
        given, so callers pass it already normalised; `text_hash` is derived here.
        """
        now = utc_iso(now)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO recordings (
                    recording_id, corpus_id, speaker_id, text, text_hash, audio_sha256, audio_path,
                    duration, quality_score, validated_at, status, first_seen, last_seen, removed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                ON CONFLICT(recording_id) DO UPDATE SET
                    corpus_id = excluded.corpus_id,
                    speaker_id = excluded.speaker_id,
                    text = excluded.text,
                    text_hash = excluded.text_hash,
                    audio_sha256 = excluded.audio_sha256,
                    audio_path = excluded.audio_path,
                    duration = excluded.duration,
                    quality_score = excluded.quality_score,
                    validated_at = excluded.validated_at,
                    status = 'active',
                    last_seen = excluded.last_seen,
                    removed_at = NULL
                """,
                (
                    recording_id, corpus_id, speaker_id, text, text_hash(text), audio_sha256, audio_path,
                    duration, quality_score, validated_at, now, now,
                ),
            )

    def mark_seen(self, ids, now=None):
        """Bump `last_seen` on active rows that the upstream listing still contains."""
        now = utc_iso(now)
        with self._conn:
            for batch in _batches(ids):
                marks = ",".join("?" * len(batch))
                self._conn.execute(
                    f"UPDATE recordings SET last_seen = ? WHERE status = 'active' AND recording_id IN ({marks})",
                    [now, *batch],
                )

    def mark_removed(self, ids, now=None):
        """Tombstone active rows; returns how many were tombstoned.

        Erases everything personal or content-bearing (text, text hash, audio
        hash and source path, speaker, duration, score). Keeps id, corpus,
        status and the first_seen/last_seen/removed_at timestamps. Unknown and
        already-removed ids are ignored, so a retry is harmless.
        """
        now = utc_iso(now)
        removed = 0
        with self._conn:
            for batch in _batches(ids):
                marks = ",".join("?" * len(batch))
                cursor = self._conn.execute(
                    f"""
                    UPDATE recordings SET
                        status = 'removed', removed_at = ?,
                        speaker_id = NULL, text = NULL, text_hash = NULL, audio_sha256 = NULL,
                        audio_path = NULL, duration = NULL, quality_score = NULL, validated_at = NULL
                    WHERE status = 'active' AND recording_id IN ({marks})
                    """,
                    [now, *batch],
                )
                removed += cursor.rowcount
        return removed

    def get(self, recording_id):
        row = self._conn.execute("SELECT * FROM recordings WHERE recording_id = ?", (recording_id,)).fetchone()
        return dict(row) if row else None

    def active_rows(self, corpus_id=None):
        """Active rows as dicts, ordered by recording_id, optionally for one corpus."""
        if corpus_id is None:
            cursor = self._conn.execute("SELECT * FROM recordings WHERE status = 'active' ORDER BY recording_id")
        else:
            cursor = self._conn.execute(
                "SELECT * FROM recordings WHERE status = 'active' AND corpus_id = ? ORDER BY recording_id", (corpus_id,)
            )
        return [dict(row) for row in cursor]

    def counts(self, corpus_id=None):
        """{'active': n, 'removed': n}, optionally for one corpus."""
        where, params = ("WHERE corpus_id = ?", (corpus_id,)) if corpus_id is not None else ("", ())
        found = dict(self._conn.execute(f"SELECT status, COUNT(*) FROM recordings {where} GROUP BY status", params).fetchall())
        return {"active": found.get("active", 0), "removed": found.get("removed", 0)}

    def audio_hash_in_use(self, sha, exclude_ids=()):
        """True when an active row outside `exclude_ids` references this audio hash.

        Identical audio shares one file, so a file may only be deleted once no
        other active recording points at it.
        """
        exclude = set(exclude_ids)
        rows = self._conn.execute(
            "SELECT recording_id FROM recordings WHERE status = 'active' AND audio_sha256 = ?", (sha,)
        )
        return any(row[0] not in exclude for row in rows)
