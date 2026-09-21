"""SQLite ledger of the crowd-source-voice recordings this installation has synced.

The ledger, not a snapshot directory, is the source of truth for training data:
`dataset sync` keeps it in step with the upstream export, and dataset builds
read only its active rows. csv hard-deletes recordings without telling anyone,
so a recording that vanishes upstream is tombstoned here and everything
personal about it (text, audio hash, speaker) is erased on the spot. The
tombstone itself stays, so recording ids are never reused and lineage records
that mention a removed recording still resolve.

A recording whose audio sync had to reject (length outside the training range,
or unconvertible) is remembered in a separate `rejected` table so that the next
sync does not download and convert it again. That table holds no text, speaker
or duration: only the id, the corpus, the reason, csv's file path and when it
was first seen. It is not part of `recordings`, so it never counts as an
active row and is never tombstoned.

Only the pseudonymous speaker id is ever stored, never a raw csv user id.
"""

import hashlib
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

LEDGER_FILENAME = "ledger.sqlite"

# (version, statements) in order. A new file gets all of them; an older file gets
# the ones above the version it carries. Every statement is idempotent, so
# running a migration twice changes nothing. Statements run one by one because
# `executescript` would commit the migration transaction.
_MIGRATIONS = (
    (1, (
        """
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
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
        )
        """,
        "CREATE INDEX IF NOT EXISTS recordings_corpus_status ON recordings (corpus_id, status)",
        "CREATE INDEX IF NOT EXISTS recordings_audio_sha256 ON recordings (audio_sha256)",
        "CREATE INDEX IF NOT EXISTS recordings_speaker_id ON recordings (speaker_id)",
    )),
    (2, (
        """
        CREATE TABLE IF NOT EXISTS rejected (
            recording_id INTEGER PRIMARY KEY,
            corpus_id    INTEGER NOT NULL,
            reason       TEXT NOT NULL,
            audio_path   TEXT,
            first_seen   TEXT NOT NULL
        )
        """,
    )),
)
SCHEMA_VERSION = _MIGRATIONS[-1][0]

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
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Zero freed pages so a tombstoned row's text does not linger in the file.
            self._conn.execute("PRAGMA secure_delete=ON")
            self._migrate()
        except Exception:
            self._conn.close()
            raise

    def _stored_version(self):
        """The schema version recorded in the file, or None when it has none yet (a new file)."""
        if not self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone():
            return None
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return int(row[0]) if row else None

    def _refuse_newer(self, found):
        if found is not None and found > SCHEMA_VERSION:
            raise LedgerError(f"Ledger {self.path} has schema version {found}; this build understands {SCHEMA_VERSION}")

    def _migrate(self):
        """Create a new file's schema, or upgrade an older file, in one transaction.

        A file from a newer build is refused before anything is written. The
        version is read again once the write lock is held, so two processes
        opening an old ledger at the same time take turns and the second finds
        nothing left to do. Python opens no transaction for DDL by itself, hence
        the explicit BEGIN; a failure rolls the whole upgrade back.
        """
        found = self._stored_version()
        self._refuse_newer(found)
        if found == SCHEMA_VERSION:
            return
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            found = self._stored_version()
            self._refuse_newer(found)
            for version, statements in _MIGRATIONS:
                if found is None or version > found:
                    for statement in statements:
                        self._conn.execute(statement)
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

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
        A recording that becomes active also leaves `rejected`, in the same
        transaction.
        """
        now = utc_iso(now)
        with self._conn:
            self._conn.execute("DELETE FROM rejected WHERE recording_id = ?", (recording_id,))
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

    def reject(self, recording_id, corpus_id, reason, audio_path, now=None):
        """Remember that the audio at `audio_path` cannot be used, so sync will not fetch it again.

        Upsert: `first_seen` is kept, the other fields follow the latest
        attempt. Nothing else is stored (no text, speaker or duration); the row
        only saves a pointless download. It is separate from `recordings`, so
        it is never counted as active and never tombstoned.
        """
        now = utc_iso(now)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO rejected (recording_id, corpus_id, reason, audio_path, first_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(recording_id) DO UPDATE SET
                    corpus_id = excluded.corpus_id,
                    reason = excluded.reason,
                    audio_path = excluded.audio_path
                """,
                (recording_id, corpus_id, reason, audio_path, now),
            )

    def rejected(self, recording_id):
        """The rejection row for this recording as a dict, or None."""
        row = self._conn.execute("SELECT * FROM rejected WHERE recording_id = ?", (recording_id,)).fetchone()
        return dict(row) if row else None

    def rejected_ids(self, corpus_id=None):
        """Ids of rejected recordings, ascending, optionally for one corpus."""
        if corpus_id is None:
            cursor = self._conn.execute("SELECT recording_id FROM rejected ORDER BY recording_id")
        else:
            cursor = self._conn.execute(
                "SELECT recording_id FROM rejected WHERE corpus_id = ? ORDER BY recording_id", (corpus_id,)
            )
        return [row[0] for row in cursor]

    def rejected_count(self, corpus_id=None):
        """How many recordings are rejected. Kept out of `counts()`, whose exact shape callers compare."""
        if corpus_id is None:
            return self._conn.execute("SELECT COUNT(*) FROM rejected").fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM rejected WHERE corpus_id = ?", (corpus_id,)).fetchone()[0]

    def clear_rejected(self, ids):
        """Forget rejections; returns how many rows were deleted. Unknown ids are ignored."""
        cleared = 0
        with self._conn:
            for batch in _batches(ids):
                marks = ",".join("?" * len(batch))
                cleared += self._conn.execute(f"DELETE FROM rejected WHERE recording_id IN ({marks})", batch).rowcount
        return cleared

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
