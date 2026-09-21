import sqlite3
from datetime import datetime, timezone

import pytest

from auditor_stt.dataset import ledger as ledger_module
from auditor_stt.dataset.ledger import LEDGER_FILENAME, Ledger, LedgerError, ledger_path, text_hash, utc_iso

T0 = "2026-01-01T00:00:00Z"
T1 = "2026-01-02T00:00:00Z"
T2 = "2026-01-03T00:00:00Z"


def _add(ledger, recording_id, corpus_id=3, *, sha=None, speaker="spk1", text="moi maailma", now=T0):
    ledger.upsert_active(
        recording_id, corpus_id, speaker_id=speaker, text=text, audio_sha256=sha or f"sha{recording_id}",
        duration=2.5, quality_score=4.5, audio_path=f"uploads/audio/{recording_id}.wav", now=now,
    )


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / LEDGER_FILENAME) as ledger:
        yield ledger


def test_creates_file_with_schema_version_and_wal(tmp_path):
    path = ledger_path(tmp_path / "nested")
    with Ledger(path):
        pass

    assert path.name == "ledger.sqlite"
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone() == ("2",)
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    finally:
        conn.close()


def test_rejects_a_ledger_from_a_newer_schema(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    with Ledger(path):
        pass
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(LedgerError):
        Ledger(path)


# --- schema v1 -> v2 migration ------------------------------------------------

# The schema as version 1 of the code created it, written out by hand on purpose:
# the test must not follow later changes to ledger.py.
_V1_TABLES = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE recordings (
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
CREATE INDEX recordings_corpus_status ON recordings (corpus_id, status);
CREATE INDEX recordings_audio_sha256 ON recordings (audio_sha256);
CREATE INDEX recordings_speaker_id ON recordings (speaker_id);
"""


def _make_v1_ledger(path, version="1"):
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_V1_TABLES)
        conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (version,))
        conn.execute(
            "INSERT INTO recordings VALUES (1, 3, 'spk1', 'moi maailma', ?, 'sha1', 'uploads/audio/1.wav', 2.5, 4.5, "
            "NULL, 'active', ?, ?, NULL)",
            (text_hash("moi maailma"), T0, T1),
        )
        conn.execute(
            "INSERT INTO recordings VALUES (2, 3, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'removed', ?, ?, ?)",
            (T0, T0, T2),
        )
        conn.commit()
    finally:
        conn.close()


def _tables_and_version(path):
    conn = sqlite3.connect(path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        return tables, version
    finally:
        conn.close()


def test_a_version_1_ledger_is_migrated_in_place_and_keeps_its_rows(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    _make_v1_ledger(path)

    with Ledger(path) as ledger:
        assert ledger.get(1) == {
            "recording_id": 1, "corpus_id": 3, "speaker_id": "spk1", "text": "moi maailma",
            "text_hash": text_hash("moi maailma"), "audio_sha256": "sha1", "audio_path": "uploads/audio/1.wav",
            "duration": 2.5, "quality_score": 4.5, "validated_at": None, "status": "active",
            "first_seen": T0, "last_seen": T1, "removed_at": None,
        }
        assert ledger.get(2)["status"] == "removed"
        assert ledger.counts() == {"active": 1, "removed": 1}
        assert ledger.rejected_count() == 0
        ledger.reject(5, 3, "duration_out_of_range", "uploads/audio/5.wav", now=T1)

    tables, version = _tables_and_version(path)
    assert tables == {"meta", "recordings", "rejected"}
    assert version == "2"


def test_migration_is_idempotent_and_leaves_later_data_alone(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    _make_v1_ledger(path)
    with Ledger(path) as ledger:
        ledger.reject(5, 3, "duration_out_of_range", "uploads/audio/5.wav", now=T1)

    with Ledger(path) as ledger:  # already v2: a no-op that must not recreate the table
        assert ledger.rejected(5)["first_seen"] == T1
        assert ledger.counts() == {"active": 1, "removed": 1}
    # Forced back to v1 (as if an old build had touched the file), the migration re-runs harmlessly.
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with Ledger(path) as ledger:
        assert ledger.rejected(5)["first_seen"] == T1
        assert ledger.get(1)["text"] == "moi maailma"

    assert _tables_and_version(path) == ({"meta", "recordings", "rejected"}, "2")


def test_a_failing_migration_rolls_back_completely(tmp_path, monkeypatch):
    path = tmp_path / LEDGER_FILENAME
    _make_v1_ledger(path)
    v1, (version, statements) = ledger_module._MIGRATIONS
    monkeypatch.setattr(ledger_module, "_MIGRATIONS", (v1, (version, (*statements, "THIS IS NOT SQL"))))

    with pytest.raises(sqlite3.OperationalError):
        Ledger(path)

    # The table created before the bad statement is gone again, and the file still says v1.
    assert _tables_and_version(path) == ({"meta", "recordings"}, "1")
    monkeypatch.undo()
    with Ledger(path) as ledger:
        assert ledger.rejected_count() == 0
    assert _tables_and_version(path) == ({"meta", "recordings", "rejected"}, "2")


def test_a_version_3_ledger_is_still_refused_and_left_untouched(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    _make_v1_ledger(path, version="3")

    with pytest.raises(LedgerError, match="schema version 3"):
        Ledger(path)

    assert _tables_and_version(path) == ({"meta", "recordings"}, "3")


# --- rejected recordings ------------------------------------------------------


def test_rejected_table_holds_only_what_avoids_a_redownload(ledger):
    columns = [r[1] for r in ledger._conn.execute("PRAGMA table_info(rejected)")]

    assert columns == ["recording_id", "corpus_id", "reason", "audio_path", "first_seen"]


def test_reject_then_read_back(ledger):
    assert ledger.rejected(4) is None

    ledger.reject(4, 3, "duration_out_of_range", "uploads/audio/4.wav", now=T0)

    assert ledger.rejected(4) == {
        "recording_id": 4, "corpus_id": 3, "reason": "duration_out_of_range",
        "audio_path": "uploads/audio/4.wav", "first_seen": T0,
    }
    # A rejection is not a recording: no row, not counted, never a tombstone.
    assert ledger.get(4) is None
    assert ledger.counts() == {"active": 0, "removed": 0}
    assert ledger.active_rows() == []


def test_reject_upserts_keeping_first_seen(ledger):
    ledger.reject(4, 3, "duration_out_of_range", "uploads/audio/4.wav", now=T0)
    ledger.reject(4, 3, "audio_undecodable", "uploads/audio/4-new.wav", now=T2)

    assert ledger.rejected(4) == {
        "recording_id": 4, "corpus_id": 3, "reason": "audio_undecodable",
        "audio_path": "uploads/audio/4-new.wav", "first_seen": T0,
    }
    assert ledger.rejected_count() == 1


def test_rejected_ids_and_count_can_be_scoped_to_a_corpus(ledger):
    ledger.reject(9, 3, "duration_out_of_range", "a.wav")
    ledger.reject(2, 3, "audio_undecodable", "b.wav")
    ledger.reject(5, 4, "duration_out_of_range", "c.wav")

    assert ledger.rejected_ids() == [2, 5, 9]
    assert ledger.rejected_ids(corpus_id=3) == [2, 9]
    assert ledger.rejected_ids(corpus_id=8) == []
    assert (ledger.rejected_count(), ledger.rejected_count(corpus_id=3), ledger.rejected_count(corpus_id=8)) == (3, 2, 0)


def test_clear_rejected_deletes_only_the_named_ids(ledger):
    for rid in (1, 2, 3):
        ledger.reject(rid, 3, "duration_out_of_range", f"{rid}.wav")

    assert ledger.clear_rejected([1, 3, 99]) == 2
    assert ledger.rejected_ids() == [2]
    assert ledger.clear_rejected([]) == 0
    assert ledger.clear_rejected({2}) == 1
    assert ledger.rejected_count() == 0


def test_clear_rejected_takes_more_ids_than_one_sql_statement_allows(ledger):
    for rid in range(1, 1201):
        ledger.reject(rid, 3, "duration_out_of_range", f"{rid}.wav")

    assert ledger.clear_rejected(range(1, 1201)) == 1200
    assert ledger.rejected_count() == 0


def test_a_recording_that_becomes_active_leaves_rejected(ledger):
    ledger.reject(7, 3, "duration_out_of_range", "uploads/audio/7.wav")
    ledger.reject(8, 3, "duration_out_of_range", "uploads/audio/8.wav")

    _add(ledger, 7)

    assert ledger.rejected(7) is None
    assert ledger.rejected(8) is not None
    assert ledger.get(7)["status"] == "active"


def test_rejected_survives_reopening_and_tombstoning_of_other_rows(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    with Ledger(path) as ledger:
        _add(ledger, 1)
        ledger.reject(2, 3, "audio_undecodable", "uploads/audio/2.wav", now=T1)
        ledger.mark_removed([1, 2], now=T2)

    with Ledger(path) as ledger:
        assert ledger.rejected(2)["reason"] == "audio_undecodable"
        assert ledger.get(2) is None


def test_upsert_then_get_round_trips_every_column(ledger):
    _add(ledger, 7)

    assert ledger.get(7) == {
        "recording_id": 7,
        "corpus_id": 3,
        "speaker_id": "spk1",
        "text": "moi maailma",
        "text_hash": text_hash("moi maailma"),
        "audio_sha256": "sha7",
        "audio_path": "uploads/audio/7.wav",
        "duration": 2.5,
        "quality_score": 4.5,
        "validated_at": None,
        "status": "active",
        "first_seen": T0,
        "last_seen": T0,
        "removed_at": None,
    }
    assert ledger.get(8) is None


def test_upsert_of_a_known_id_updates_in_place_and_keeps_first_seen(ledger):
    _add(ledger, 7, now=T0)
    _add(ledger, 7, text="uusi teksti", now=T1)

    row = ledger.get(7)
    assert row["text"] == "uusi teksti"
    assert row["text_hash"] == text_hash("uusi teksti")
    assert (row["first_seen"], row["last_seen"]) == (T0, T1)
    assert ledger.counts() == {"active": 1, "removed": 0}


def test_mark_removed_erases_personal_fields_and_keeps_the_tombstone(ledger):
    _add(ledger, 1)
    _add(ledger, 2)

    assert ledger.mark_removed([1], now=T1) == 1

    row = ledger.get(1)
    assert row["status"] == "removed"
    assert row["removed_at"] == T1
    assert (row["recording_id"], row["corpus_id"], row["first_seen"], row["last_seen"]) == (1, 3, T0, T0)
    for erased in ("text", "text_hash", "audio_sha256", "audio_path", "speaker_id", "duration", "quality_score"):
        assert row[erased] is None, erased
    assert ledger.get(2)["status"] == "active"
    assert ledger.counts() == {"active": 1, "removed": 1}
    assert [r["recording_id"] for r in ledger.active_rows()] == [2]


def test_erased_text_is_not_left_in_the_database_file(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    with Ledger(path) as ledger:
        _add(ledger, 1, text="ainutlaatuinen lause", speaker="spkerase")
        ledger.mark_removed([1], now=T1)

    # WAL is checkpointed on close; scan whatever is on disk.
    on_disk = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert b"ainutlaatuinen" not in on_disk
    assert b"spkerase" not in on_disk


def test_mark_removed_ignores_unknown_and_already_removed_ids(ledger):
    _add(ledger, 1)

    assert ledger.mark_removed([1, 99], now=T1) == 1
    assert ledger.mark_removed([1], now=T2) == 0
    assert ledger.get(1)["removed_at"] == T1
    assert ledger.get(99) is None


def test_upsert_revives_a_tombstone(ledger):
    _add(ledger, 1, now=T0)
    ledger.mark_removed([1], now=T1)
    _add(ledger, 1, now=T2)

    row = ledger.get(1)
    assert row["status"] == "active"
    assert row["removed_at"] is None
    assert row["first_seen"] == T0
    assert row["text"] == "moi maailma"


def test_mark_seen_only_touches_active_rows(ledger):
    _add(ledger, 1)
    _add(ledger, 2)
    ledger.mark_removed([2], now=T1)

    ledger.mark_seen([1, 2, 99], now=T2)

    assert ledger.get(1)["last_seen"] == T2
    assert ledger.get(2)["last_seen"] == T0
    assert ledger.get(99) is None


def test_active_rows_and_counts_can_be_scoped_to_a_corpus(ledger):
    _add(ledger, 1, corpus_id=3)
    _add(ledger, 2, corpus_id=3)
    _add(ledger, 3, corpus_id=4)
    ledger.mark_removed([2], now=T1)

    assert [r["recording_id"] for r in ledger.active_rows(corpus_id=3)] == [1]
    assert [r["recording_id"] for r in ledger.active_rows()] == [1, 3]
    assert ledger.counts(corpus_id=3) == {"active": 1, "removed": 1}
    assert ledger.counts() == {"active": 2, "removed": 1}


def test_audio_hash_in_use(ledger):
    _add(ledger, 1, sha="shared")
    _add(ledger, 2, sha="shared")
    _add(ledger, 3, sha="alone")

    assert ledger.audio_hash_in_use("shared")
    assert ledger.audio_hash_in_use("shared", exclude_ids=[1])
    assert not ledger.audio_hash_in_use("shared", exclude_ids=[1, 2])
    assert ledger.audio_hash_in_use("alone")
    assert not ledger.audio_hash_in_use("alone", exclude_ids=[3])
    assert not ledger.audio_hash_in_use("unknown")

    ledger.mark_removed([1, 2], now=T1)
    assert not ledger.audio_hash_in_use("shared")


def test_many_ids_in_one_call_exceed_no_sql_parameter_limit(ledger):
    for recording_id in range(1, 1201):
        _add(ledger, recording_id)

    assert ledger.mark_removed(range(1, 1201), now=T1) == 1200
    ledger.mark_seen(range(1, 1201), now=T2)
    assert ledger.counts() == {"active": 0, "removed": 1200}


def test_state_survives_reopening(tmp_path):
    path = tmp_path / LEDGER_FILENAME
    with Ledger(path) as ledger:
        _add(ledger, 1)
    with Ledger(path) as ledger:
        assert ledger.get(1)["text"] == "moi maailma"


def test_utc_iso_accepts_datetimes_strings_and_none():
    assert utc_iso(datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)) == "2026-01-02T03:04:05Z"
    assert utc_iso(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05Z"
    assert utc_iso("2026-01-02T03:04:05Z") == "2026-01-02T03:04:05Z"
    assert utc_iso().endswith("Z")


def test_text_hash_is_stable_across_unicode_forms():
    assert text_hash("hyvää") == text_hash("hyvää")
    assert text_hash(None) is None
