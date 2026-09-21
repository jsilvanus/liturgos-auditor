import hashlib
import hmac
import io
import math
import shutil
import struct
import wave

import httpx
import pytest

from auditor_stt.cli import main
from auditor_stt.dataset.audio import AudioNormalizeError, AudioNormalizeTimeout, audio_path, normalize_audio
from auditor_stt.dataset.client import CrowdSourceVoiceClient
from auditor_stt.dataset.ledger import Ledger, ledger_path
from auditor_stt.dataset.pull import ExportInconsistentError, UnsupportedCorpusTypeError
from auditor_stt.dataset.sync import MassRemovalError, SyncReport, run_sync, sync_corpus

BASE_URL = "https://csv.example.org"
T0 = "2026-01-01T00:00:00Z"
T1 = "2026-02-01T00:00:00Z"

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def make_wav(seconds, seed=0, rate=1000):
    """A valid mono WAV of the given length; a low sample rate keeps long clips small, `seed` makes the bytes distinct."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", seed) * round(seconds * rate))
    return buf.getvalue()


def passthrough(data, source_name):
    return data


class FakeCsv:
    """crowd-source-voice as sync sees it: the export, the manifest and the audio route, nothing else.

    `v2` rows carry `recording_id`/`audio_url`; legacy rows are named by
    position and are paired through the manifest. Any other request fails the test.
    """

    def __init__(self, *, v2=True, corpus_type="text"):
        self.v2 = v2
        self.corpus_type = corpus_type
        self.recordings = {}
        self.requests = []
        self.fail_audio = set()
        self.manifest_hook = None
        self.manifest_calls = 0

    def add(self, rid, text="moi maailma", seconds=2.0, speaker="spkhash1", score=4.5, reported_duration=None, audio=None):
        self.recordings[rid] = {
            "text": text,
            "speaker": speaker,
            "score": score,
            "path": f"uploads/audio/{rid}.wav",
            "audio": audio if audio is not None else make_wav(seconds, seed=rid),
            "duration": seconds if reported_duration is None else reported_duration,
        }

    def remove(self, rid):
        del self.recordings[rid]

    def downloads(self):
        return [path for path in self.requests if path.startswith("/uploads/")]

    def calls(self, path):
        return self.requests.count(path)

    def _ordered(self):
        return [(f"{i:04d}.wav", rid, rec) for i, (rid, rec) in enumerate(sorted(self.recordings.items()), start=1)]

    def _export(self):
        rows = []
        for name, rid, rec in self._ordered():
            row = {
                "file": name,
                "original_path": "/" + rec["path"],
                "text": rec["text"],
                "duration": rec["duration"],
                "quality_score": rec["score"],
                "validation_count": 2,
            }
            if self.v2:
                row.update(recording_id=rid, audio_url="/" + rec["path"], speaker_id=rec["speaker"])
            elif rec["speaker"] is not None:
                row["speaker_id"] = rec["speaker"]
            rows.append(row)
        corpus = {"id": 3, "name": "Sunday sermons", "language": "fi", "type": self.corpus_type}
        return {"corpus": corpus, "total_recordings": len(rows), "recordings": rows}

    def _manifest(self):
        files = [
            {"id": rid, "source_path": rec["path"], "export_name": name, "text": rec["text"]}
            for name, rid, rec in self._ordered()
        ]
        return {"total": len(files), "files": files}

    def __call__(self, request):
        path = request.url.path
        self.requests.append(path)
        assert not path.startswith("/api/admin"), "sync must never touch the admin API"
        if path == "/api/export":
            assert "include_all" not in request.url.params
            if not self.recordings:
                return httpx.Response(404, json={"error": "No recordings found for export"})
            return httpx.Response(200, json=self._export())
        if path == "/api/export/manifest":
            self.manifest_calls += 1
            if not self.recordings:
                return httpx.Response(404, json={"error": "No recordings found for export"})
            manifest = self._manifest()
            if self.manifest_hook:
                manifest = self.manifest_hook(self.manifest_calls, manifest)
            return httpx.Response(200, json=manifest)
        if path.startswith("/uploads/"):
            for rid, rec in self.recordings.items():
                if "/" + rec["path"] == path:
                    if rid in self.fail_audio:
                        return httpx.Response(500)
                    return httpx.Response(200, content=rec["audio"])
            return httpx.Response(404)
        raise AssertionError(f"unexpected request: {request.url}")


@pytest.fixture
def fake():
    return FakeCsv()


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def ledger(data_dir):
    with Ledger(ledger_path(data_dir)) as ledger:
        yield ledger


def sync(fake, ledger, data_dir, **kwargs):
    kwargs.setdefault("normalizer", passthrough)
    client = CrowdSourceVoiceClient(BASE_URL, "test-token", transport=httpx.MockTransport(fake))
    with client:
        return sync_corpus(client, ledger, 3, data_dir, **kwargs)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audio_files(data_dir):
    audio_dir = data_dir / "audio"
    return sorted(p.name for p in audio_dir.iterdir()) if audio_dir.exists() else []


def counts(report):
    return (report.added, report.updated, report.removed, report.unchanged)


# --- initial sync -----------------------------------------------------------


def test_initial_sync_with_stable_recording_ids(fake, ledger, data_dir):
    fake.add(11, text="  moi\nmaailma ", seconds=2.0, speaker="spkhash1")
    fake.add(12, text="hyvää huomenta", seconds=3.5, speaker="spkhash2", score=4.2)

    report = sync(fake, ledger, data_dir, now=T0)

    assert counts(report) == (2, 0, 0, 0)
    assert report.skipped == {}
    wav = fake.recordings[11]["audio"]
    assert audio_path(data_dir, sha(wav)).read_bytes() == wav
    row = ledger.get(11)
    assert row["corpus_id"] == 3
    assert row["text"] == "moi maailma"
    assert row["speaker_id"] == "spkhash1"
    assert row["audio_sha256"] == sha(wav)
    assert row["audio_path"] == "uploads/audio/11.wav"
    assert row["duration"] == pytest.approx(2.0)
    assert row["quality_score"] == 4.5
    assert (row["status"], row["first_seen"], row["last_seen"]) == ("active", T0, T0)
    assert ledger.get(12)["quality_score"] == 4.2
    # Stable ids make the manifest, and its pairing race, unnecessary.
    assert fake.calls("/api/export/manifest") == 0


def test_initial_sync_of_a_legacy_export_pairs_rows_through_the_manifest(data_dir, ledger):
    fake = FakeCsv(v2=False)
    fake.add(5, text="ensimmäinen", seconds=2.0, speaker="spkhash1")
    fake.add(9, text="toinen", seconds=3.0, speaker=None)

    report = sync(fake, ledger, data_dir, now=T0)

    assert counts(report) == (2, 0, 0, 0)
    assert [r["recording_id"] for r in ledger.active_rows()] == [5, 9]
    assert ledger.get(5)["text"] == "ensimmäinen"
    assert ledger.get(5)["speaker_id"] == "spkhash1"
    assert ledger.get(9)["speaker_id"] is None
    assert sha(fake.recordings[9]["audio"]) in "".join(audio_files(data_dir))
    assert fake.calls("/api/export/manifest") == 1
    assert sorted(fake.downloads()) == ["/uploads/audio/5.wav", "/uploads/audio/9.wav"]


def test_upgrading_csv_from_legacy_to_stable_ids_redownloads_nothing(data_dir, ledger):
    fake = FakeCsv(v2=False)
    fake.add(5)
    fake.add(9, seconds=3.0)
    sync(fake, ledger, data_dir, now=T0)
    before = len(fake.downloads())

    fake.v2 = True
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (0, 0, 0, 2)
    assert len(fake.downloads()) == before


def test_empty_listing_is_a_valid_empty_sync_even_for_a_non_text_corpus(ledger, data_dir):
    fake = FakeCsv(corpus_type="music")

    report = sync(fake, ledger, data_dir)

    assert counts(report) == (0, 0, 0, 0)
    assert fake.calls("/api/export/manifest") == 0
    assert audio_files(data_dir) == []


def test_non_text_corpus_is_rejected_before_any_download(ledger, data_dir):
    fake = FakeCsv(corpus_type="music")
    fake.add(1)

    with pytest.raises(UnsupportedCorpusTypeError):
        sync(fake, ledger, data_dir)

    assert fake.downloads() == []
    assert fake.calls("/api/export/manifest") == 0
    assert ledger.counts() == {"active": 0, "removed": 0}


# --- incremental behaviour --------------------------------------------------


def test_rerun_with_no_upstream_change_is_all_unchanged_and_downloads_nothing(fake, ledger, data_dir):
    fake.add(1)
    fake.add(2, seconds=3.0)
    sync(fake, ledger, data_dir, now=T0)
    downloads = len(fake.downloads())

    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (0, 0, 0, 2)
    assert report.skipped == {}
    assert len(fake.downloads()) == downloads
    row = ledger.get(1)
    assert (row["first_seen"], row["last_seen"]) == (T0, T1)


def test_changed_text_and_score_update_in_place_without_redownloading(fake, ledger, data_dir):
    fake.add(1, text="vanha teksti")
    fake.add(2)
    sync(fake, ledger, data_dir, now=T0)
    original_sha = ledger.get(1)["audio_sha256"]
    downloads = len(fake.downloads())

    fake.recordings[1]["text"] = "uusi teksti"
    fake.recordings[1]["score"] = 3.0
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (0, 1, 0, 1)
    assert len(fake.downloads()) == downloads
    row = ledger.get(1)
    assert (row["text"], row["quality_score"]) == ("uusi teksti", 3.0)
    assert row["audio_sha256"] == original_sha
    assert (row["first_seen"], row["last_seen"]) == (T0, T1)


def test_changed_audio_path_redownloads_and_deletes_the_superseded_file(fake, ledger, data_dir):
    fake.add(1)
    sync(fake, ledger, data_dir, now=T0)
    old_sha = ledger.get(1)["audio_sha256"]

    fake.recordings[1]["path"] = "uploads/audio/1-rerecorded.wav"
    fake.recordings[1]["audio"] = make_wav(2.0, seed=500)
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (0, 1, 0, 0)
    new_sha = ledger.get(1)["audio_sha256"]
    assert new_sha == sha(fake.recordings[1]["audio"])
    assert audio_files(data_dir) == [f"{new_sha}.wav"]
    assert old_sha != new_sha


def test_a_deleted_audio_file_is_fetched_again(fake, ledger, data_dir):
    fake.add(1)
    sync(fake, ledger, data_dir)
    audio_path(data_dir, ledger.get(1)["audio_sha256"]).unlink()

    report = sync(fake, ledger, data_dir)

    assert counts(report) == (0, 1, 0, 0)
    assert len(audio_files(data_dir)) == 1


# --- removal ----------------------------------------------------------------


def test_removed_recording_is_tombstoned_and_its_audio_deleted_only_when_unreferenced(fake, ledger, data_dir):
    shared = make_wav(2.0, seed=99)
    fake.add(1, audio=shared, text="ensimmäinen", speaker="spkA")
    fake.add(2, audio=shared, text="toinen", speaker="spkB")
    fake.add(3, seconds=3.0)
    sync(fake, ledger, data_dir, now=T0)
    assert len(audio_files(data_dir)) == 2  # 1 and 2 share one file

    fake.remove(1)
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (0, 0, 1, 2)
    assert audio_path(data_dir, sha(shared)).exists()  # recording 2 still needs it
    row = ledger.get(1)
    assert (row["status"], row["removed_at"]) == ("removed", T1)
    assert (row["text"], row["text_hash"], row["audio_sha256"], row["speaker_id"]) == (None, None, None, None)

    fake.remove(2)
    report = sync(fake, ledger, data_dir, now="2026-03-01T00:00:00Z")

    assert counts(report) == (0, 0, 1, 1)
    assert not audio_path(data_dir, sha(shared)).exists()
    assert audio_files(data_dir) == [f"{ledger.get(3)['audio_sha256']}.wav"]
    assert ledger.counts() == {"active": 1, "removed": 2}


def test_removal_of_one_corpus_does_not_touch_another_corpus_that_shares_audio(fake, ledger, data_dir):
    shared = make_wav(2.0, seed=99)
    fake.add(1, audio=shared)
    sync(fake, ledger, data_dir)
    ledger.upsert_active(
        500, 4, speaker_id=None, text="toisesta korpuksesta", audio_sha256=sha(shared), duration=2.0, quality_score=None,
    )

    fake.remove(1)
    sync(fake, ledger, data_dir)

    assert audio_path(data_dir, sha(shared)).exists()
    assert ledger.get(500)["status"] == "active"


def test_removed_recording_that_reappears_is_downloaded_again(fake, ledger, data_dir):
    fake.add(1)
    sync(fake, ledger, data_dir, now=T0)
    fake.remove(1)
    sync(fake, ledger, data_dir, now=T1)
    assert audio_files(data_dir) == []

    fake.add(1)
    report = sync(fake, ledger, data_dir, now="2026-03-01T00:00:00Z")

    assert counts(report) == (1, 0, 0, 0)
    row = ledger.get(1)
    assert (row["status"], row["removed_at"], row["first_seen"]) == ("active", None, T0)
    assert len(audio_files(data_dir)) == 1


@pytest.mark.parametrize(
    "total, remaining, refused",
    [
        (4, 0, False),   # small corpus: wiping 4 rows is allowed
        (5, 0, False),   # exactly 5 is not "more than 5"
        (6, 0, True),
        (12, 6, False),  # exactly half is not "more than 50%"
        (12, 5, True),
    ],
)
def test_mass_removal_guard_thresholds(fake, ledger, data_dir, total, remaining, refused):
    for rid in range(1, total + 1):
        fake.add(rid)
    sync(fake, ledger, data_dir)
    for rid in range(remaining + 1, total + 1):
        fake.remove(rid)

    if refused:
        with pytest.raises(MassRemovalError):
            sync(fake, ledger, data_dir)
        assert ledger.counts() == {"active": total, "removed": 0}
    else:
        report = sync(fake, ledger, data_dir)
        assert report.removed == total - remaining


def test_mass_removal_refuses_without_changing_anything_then_passes_with_the_flag(fake, ledger, data_dir):
    for rid in range(1, 11):
        fake.add(rid, seconds=2.0 + rid / 10)
    sync(fake, ledger, data_dir, now=T0)
    files_before = audio_files(data_dir)
    for rid in range(3, 11):
        fake.remove(rid)

    with pytest.raises(MassRemovalError, match="8 of 10"):
        sync(fake, ledger, data_dir, now=T1)
    assert ledger.counts() == {"active": 10, "removed": 0}
    assert audio_files(data_dir) == files_before

    report = sync(fake, ledger, data_dir, now=T1, allow_mass_removal=True)

    assert counts(report) == (0, 0, 8, 2)
    assert ledger.counts() == {"active": 2, "removed": 8}
    assert len(audio_files(data_dir)) == 2


def test_an_empty_export_cannot_wipe_a_large_ledger(fake, ledger, data_dir):
    for rid in range(1, 9):
        fake.add(rid)
    sync(fake, ledger, data_dir)
    fake.recordings.clear()  # csv answers 404 "No recordings found"

    with pytest.raises(MassRemovalError):
        sync(fake, ledger, data_dir)
    assert ledger.counts()["active"] == 8

    report = sync(fake, ledger, data_dir, allow_mass_removal=True)
    assert report.removed == 8
    assert audio_files(data_dir) == []


def test_an_empty_export_tombstones_a_small_ledger(fake, ledger, data_dir):
    fake.add(1)
    fake.add(2, seconds=3.0)
    sync(fake, ledger, data_dir)
    fake.recordings.clear()

    report = sync(fake, ledger, data_dir)

    assert counts(report) == (0, 0, 2, 0)
    assert audio_files(data_dir) == []


# --- speaker ids ------------------------------------------------------------


def _expected_hash(salt, raw):
    return hmac.new(salt.encode(), str(raw).encode(), hashlib.sha256).hexdigest()[:24]


def test_raw_speaker_ids_are_hashed_with_the_salt(fake, ledger, data_dir):
    fake.add(1, speaker=42)
    fake.add(2, speaker="43", seconds=3.0)

    report = sync(fake, ledger, data_dir, speaker_salt="pepper", now=T0)

    assert report.skipped == {}
    assert ledger.get(1)["speaker_id"] == _expected_hash("pepper", 42)
    assert ledger.get(2)["speaker_id"] == _expected_hash("pepper", 43)
    assert len(ledger.get(1)["speaker_id"]) == 24
    # Hashing is deterministic, so the next run sees nothing changed.
    assert counts(sync(fake, ledger, data_dir, speaker_salt="pepper", now=T1)) == (0, 0, 0, 2)


def test_raw_speaker_ids_are_dropped_and_counted_without_a_salt(fake, ledger, data_dir):
    fake.add(1, speaker=42)
    fake.add(2, speaker="43", seconds=3.0)
    fake.add(3, speaker="spkhash1", seconds=2.5)

    report = sync(fake, ledger, data_dir)

    assert counts(report) == (3, 0, 0, 0)
    assert report.skipped == {"raw_speaker_id_dropped": 2}
    assert [ledger.get(i)["speaker_id"] for i in (1, 2, 3)] == [None, None, "spkhash1"]


def test_pseudonymous_speaker_ids_pass_through_even_when_a_salt_is_set(fake, ledger, data_dir):
    pseudonym = "a3f9c2d1e4b5a6c7d8e9f0a1"
    fake.add(1, speaker=pseudonym)
    fake.add(2, speaker="spkhash1", seconds=3.0)
    fake.add(3, speaker=None, seconds=2.5)  # anonymised recording

    report = sync(fake, ledger, data_dir, speaker_salt="pepper")

    assert report.skipped == {}
    assert [ledger.get(i)["speaker_id"] for i in (1, 2, 3)] == [pseudonym, "spkhash1", None]


def test_all_digit_pseudonym_is_not_mistaken_for_a_raw_user_id(fake, ledger, data_dir):
    # 24 hex chars that all happen to be decimal digits: still csv's pseudonym, not a raw id.
    pseudonym = "123456789012345678901234"
    fake.add(1, speaker=pseudonym)

    for salt in ("pepper", None):
        report = sync(fake, ledger, data_dir, speaker_salt=salt, now=T0)
        assert report.skipped == {}
        assert ledger.get(1)["speaker_id"] == pseudonym


def test_email_like_speaker_ids_are_never_stored(fake, ledger, data_dir):
    fake.add(1, speaker="someone@example.org")

    report = sync(fake, ledger, data_dir, speaker_salt="pepper")

    assert report.skipped == {"unusable_speaker_id_dropped": 1}
    assert ledger.get(1)["speaker_id"] is None


# --- audio gate and failures ------------------------------------------------


def test_duration_gate_uses_the_real_audio_length_not_the_exports(fake, ledger, data_dir):
    fake.add(1, seconds=45.0, reported_duration=2.0)   # export claims a fine length; the audio is too long
    fake.add(2, seconds=3.0, reported_duration=100.0)  # export claims too long; the audio is fine
    fake.add(3, seconds=0.2, reported_duration=2.0)    # too short
    fake.add(4, seconds=0.5)                           # exactly the lower bound
    fake.add(5, seconds=30.0)                          # exactly the upper bound

    report = sync(fake, ledger, data_dir)

    assert counts(report) == (3, 0, 0, 0)
    assert report.skipped == {"duration_out_of_range": 2}
    assert sorted(r["recording_id"] for r in ledger.active_rows()) == [2, 4, 5]
    assert len(audio_files(data_dir)) == 3  # rejected audio is never stored
    assert ledger.get(2)["duration"] == pytest.approx(3.0)


def test_one_failed_download_does_not_abort_or_tombstone_anything(fake, ledger, data_dir):
    for rid in (1, 2, 3):
        fake.add(rid, seconds=2.0 + rid)
    fake.fail_audio.add(2)

    report = sync(fake, ledger, data_dir, now=T0)

    assert counts(report) == (2, 0, 0, 0)
    assert report.skipped == {"download_failed": 1}
    assert ledger.get(2) is None

    fake.fail_audio.clear()
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (1, 0, 0, 2)
    assert report.skipped == {}


def test_failed_download_of_a_known_recording_leaves_its_row_untouched(fake, ledger, data_dir):
    fake.add(1)
    sync(fake, ledger, data_dir, now=T0)
    before = ledger.get(1)

    fake.recordings[1]["path"] = "uploads/audio/1-moved.wav"
    fake.fail_audio.add(1)
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (0, 0, 0, 0)
    assert report.skipped == {"download_failed": 1}
    after = ledger.get(1)
    assert after["status"] == "active"
    assert {k: v for k, v in after.items() if k != "last_seen"} == {k: v for k, v in before.items() if k != "last_seen"}
    assert audio_path(data_dir, before["audio_sha256"]).exists()


def test_undecodable_audio_is_skipped_and_counted(fake, ledger, data_dir):
    fake.add(1)
    fake.add(2, seconds=3.0)
    fake.add(3, seconds=4.0)

    def normalizer(data, source_name):
        if source_name.endswith("/1.wav"):
            raise AudioNormalizeError("ffmpeg exited with 1")
        if source_name.endswith("/2.wav"):
            return b"not a wav file"
        return data

    report = sync(fake, ledger, data_dir, normalizer=normalizer)

    assert counts(report) == (1, 0, 0, 0)
    assert report.skipped == {"audio_undecodable": 2}
    assert [r["recording_id"] for r in ledger.active_rows()] == [3]


def test_recordings_without_text_are_skipped(fake, ledger, data_dir):
    fake.add(1, text="   ")
    fake.add(2, seconds=3.0)

    report = sync(fake, ledger, data_dir)

    assert counts(report) == (1, 0, 0, 0)
    assert report.skipped == {"empty_text": 1}
    assert fake.downloads() == ["/uploads/audio/2.wav"]


# --- rejected recordings ----------------------------------------------------


def test_a_rejected_recording_is_not_downloaded_again(fake, ledger, data_dir):
    fake.add(1, seconds=45.0)  # too long
    fake.add(2, seconds=0.2)   # too short
    fake.add(3, seconds=3.0)
    report = sync(fake, ledger, data_dir, now=T0)

    assert counts(report) == (1, 0, 0, 0)
    assert report.skipped == {"duration_out_of_range": 2}
    assert sorted(fake.downloads()) == [f"/uploads/audio/{rid}.wav" for rid in (1, 2, 3)]
    assert ledger.rejected(1) == {
        "recording_id": 1, "corpus_id": 3, "reason": "duration_out_of_range",
        "audio_path": "uploads/audio/1.wav", "first_seen": T0,
    }
    assert ledger.rejected_ids() == [1, 2]

    fake.requests.clear()
    report = sync(fake, ledger, data_dir, now=T1)

    assert fake.downloads() == []
    assert counts(report) == (0, 0, 0, 1)
    assert report.skipped == {"previously_rejected": 2}
    assert "previously_rejected=2" in str(report)
    assert ledger.rejected(1)["first_seen"] == T0
    assert ledger.get(1) is None  # a rejection is never a ledger row, let alone a tombstone


def test_unconvertible_audio_is_rejected_once_and_not_converted_again(fake, ledger, data_dir):
    fake.add(1)
    fake.add(2, seconds=3.0)
    converted = []

    def normalizer(data, source_name):
        converted.append(source_name)
        if source_name.endswith("/1.wav"):
            raise AudioNormalizeError("ffmpeg exited with 1")
        return data

    sync(fake, ledger, data_dir, normalizer=normalizer, now=T0)
    assert ledger.rejected(1)["reason"] == "audio_undecodable"
    fake.requests.clear()
    converted.clear()

    report = sync(fake, ledger, data_dir, normalizer=normalizer, now=T1)

    assert fake.downloads() == [] and converted == []
    assert report.skipped == {"previously_rejected": 1}


def test_a_rejected_recording_with_a_new_audio_path_is_tried_again_and_updated(fake, ledger, data_dir):
    fake.add(1, seconds=45.0)
    sync(fake, ledger, data_dir, now=T0)
    fake.requests.clear()

    fake.recordings[1]["path"] = "uploads/audio/1-again.wav"  # re-recorded, but still too long
    report = sync(fake, ledger, data_dir, now=T1)

    assert fake.downloads() == ["/uploads/audio/1-again.wav"]
    assert report.skipped == {"duration_out_of_range": 1}
    assert ledger.rejected(1) == {
        "recording_id": 1, "corpus_id": 3, "reason": "duration_out_of_range",
        "audio_path": "uploads/audio/1-again.wav", "first_seen": T0,
    }

    fake.requests.clear()
    report = sync(fake, ledger, data_dir, now="2026-03-01T00:00:00Z")

    assert fake.downloads() == []
    assert report.skipped == {"previously_rejected": 1}


def test_a_rejected_recording_that_later_succeeds_leaves_the_table(fake, ledger, data_dir):
    fake.add(1, seconds=45.0)
    sync(fake, ledger, data_dir, now=T0)
    assert ledger.rejected(1) is not None

    fake.recordings[1]["path"] = "uploads/audio/1-cut.wav"
    fake.recordings[1]["audio"] = make_wav(3.0, seed=77)
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (1, 0, 0, 0)
    assert report.skipped == {}
    assert ledger.rejected(1) is None
    assert ledger.rejected_count() == 0
    row = ledger.get(1)
    assert (row["status"], row["audio_path"], row["first_seen"]) == ("active", "uploads/audio/1-cut.wav", T1)


def test_an_active_recording_whose_new_audio_is_rejected_keeps_its_row_and_is_not_refetched(fake, ledger, data_dir):
    fake.add(1)
    sync(fake, ledger, data_dir, now=T0)
    before = ledger.get(1)

    fake.recordings[1]["path"] = "uploads/audio/1-long.wav"
    fake.recordings[1]["audio"] = make_wav(45.0, seed=5)
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (0, 0, 0, 0)
    assert report.skipped == {"duration_out_of_range": 1}
    assert {k: v for k, v in ledger.get(1).items() if k != "last_seen"} == {k: v for k, v in before.items() if k != "last_seen"}
    assert audio_path(data_dir, before["audio_sha256"]).exists()

    fake.requests.clear()
    report = sync(fake, ledger, data_dir, now="2026-03-01T00:00:00Z")

    assert fake.downloads() == []
    assert report.skipped == {"previously_rejected": 1}
    assert ledger.get(1)["status"] == "active"


def test_a_rejected_recording_that_vanishes_upstream_is_forgotten(fake, ledger, data_dir):
    fake.add(1, seconds=45.0)
    fake.add(2, seconds=3.0)
    sync(fake, ledger, data_dir, now=T0)
    ledger.reject(50, 4, "duration_out_of_range", "uploads/audio/50.wav")  # another corpus: not this sync's business

    fake.remove(1)
    report = sync(fake, ledger, data_dir, now=T1)

    assert counts(report) == (0, 0, 0, 1)
    assert ledger.rejected(1) is None
    assert ledger.get(1) is None
    assert ledger.rejected_ids() == [50]
    assert ledger.counts() == {"active": 1, "removed": 0}


def test_a_failed_download_is_transient_and_never_recorded_as_a_rejection(fake, ledger, data_dir):
    fake.add(1, seconds=45.0)
    fake.fail_audio.add(1)

    report = sync(fake, ledger, data_dir, now=T0)

    assert report.skipped == {"download_failed": 1}
    assert ledger.rejected(1) is None
    assert ledger.rejected_count() == 0

    fake.fail_audio.clear()
    report = sync(fake, ledger, data_dir, now=T1)  # the download now works, so the audio is judged for real

    assert report.skipped == {"duration_out_of_range": 1}
    assert ledger.rejected(1)["first_seen"] == T1


def test_a_failed_download_of_a_changed_path_keeps_the_old_rejection(fake, ledger, data_dir):
    fake.add(1, seconds=45.0)
    sync(fake, ledger, data_dir, now=T0)

    fake.recordings[1]["path"] = "uploads/audio/1-again.wav"
    fake.fail_audio.add(1)
    report = sync(fake, ledger, data_dir, now=T1)

    assert report.skipped == {"download_failed": 1}
    assert ledger.rejected(1)["audio_path"] == "uploads/audio/1.wav"


def test_mass_removal_guard_does_not_count_rejected_recordings_as_removals(fake, ledger, data_dir):
    for rid in range(1, 5):
        fake.add(rid)
    for rid in range(5, 15):
        fake.add(rid, seconds=0.2)  # rejected
    sync(fake, ledger, data_dir, now=T0)
    assert ledger.rejected_count() == 10
    for rid in range(5, 15):
        fake.remove(rid)

    report = sync(fake, ledger, data_dir, now=T1)  # 10 vanished ids, none of them an active row: no refusal

    assert counts(report) == (0, 0, 0, 4)
    assert ledger.counts() == {"active": 4, "removed": 0}
    assert ledger.rejected_count() == 0


def test_mass_removal_guard_ignores_rejected_rows_in_its_denominator_and_refuses_before_changing_anything(
    fake, ledger, data_dir,
):
    for rid in range(1, 11):
        fake.add(rid)
    for rid in range(11, 31):
        fake.add(rid, seconds=0.2)  # 20 rejected rows must not dilute the 10 active ones
    sync(fake, ledger, data_dir, now=T0)
    for rid in range(5, 11):
        fake.remove(rid)  # 6 of 10 active rows
    fake.remove(11)       # and one rejected one

    with pytest.raises(MassRemovalError, match="6 of 10"):
        sync(fake, ledger, data_dir, now=T1)
    assert ledger.counts() == {"active": 10, "removed": 0}
    assert ledger.rejected_count() == 20

    report = sync(fake, ledger, data_dir, now=T1, allow_mass_removal=True)

    assert report.removed == 6
    assert ledger.rejected_count() == 19
    assert ledger.counts() == {"active": 4, "removed": 6}


# --- legacy pairing race ----------------------------------------------------


def _shifted(manifest):
    # Same positional names, but 0001.wav now names a different recording than the export row it must pair with.
    files = [dict(f) for f in manifest["files"]]
    files[0].update(id=7, source_path="uploads/audio/other.wav", text="aivan eri lause")
    return {"total": len(files), "files": files}


def test_legacy_race_refetches_once_then_pairs_correctly(ledger, data_dir):
    fake = FakeCsv(v2=False)
    fake.add(1, text="ensimmäinen")
    fake.add(2, text="toinen", seconds=3.0)
    fake.manifest_hook = lambda call, manifest: _shifted(manifest) if call == 1 else manifest

    report = sync(fake, ledger, data_dir)

    assert counts(report) == (2, 0, 0, 0)
    assert (fake.calls("/api/export"), fake.calls("/api/export/manifest")) == (2, 2)
    assert "/uploads/audio/other.wav" not in fake.downloads()
    assert ledger.get(1)["audio_sha256"] == sha(fake.recordings[1]["audio"])
    assert ledger.get(7) is None


def test_legacy_race_that_persists_aborts_before_downloading_anything(ledger, data_dir):
    fake = FakeCsv(v2=False)
    fake.add(1, text="ensimmäinen")
    fake.add(2, text="toinen", seconds=3.0)
    fake.manifest_hook = lambda call, manifest: _shifted(manifest)

    with pytest.raises(ExportInconsistentError):
        sync(fake, ledger, data_dir)

    assert (fake.calls("/api/export"), fake.calls("/api/export/manifest")) == (2, 2)
    assert fake.downloads() == []
    assert ledger.counts() == {"active": 0, "removed": 0}


# --- real ffmpeg ------------------------------------------------------------


def _stereo_wav_8k(seconds=1.0):
    rate = 8000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(int(seconds * rate)):
            sample = int(8000 * math.sin(2 * math.pi * 440 * i / rate))
            frames += struct.pack("<hh", sample, -sample)
        w.writeframes(bytes(frames))
    return buf.getvalue()


@needs_ffmpeg
def test_normalize_audio_produces_16khz_mono_pcm16_deterministically():
    source = _stereo_wav_8k(1.0)

    out = normalize_audio(source, "uploads/audio/x.webm")  # a misleading extension must not matter

    with wave.open(io.BytesIO(out), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() / w.getframerate() == pytest.approx(1.0, abs=0.05)
    assert normalize_audio(source, "x.wav") == out


@needs_ffmpeg
def test_normalize_audio_rejects_garbage():
    with pytest.raises(AudioNormalizeError):
        normalize_audio(b"this is not audio at all", "x.wav")


@needs_ffmpeg
def test_sync_with_the_real_ffmpeg_normaliser(fake, ledger, data_dir):
    fake.add(1, audio=_stereo_wav_8k(1.0))

    report = sync(fake, ledger, data_dir, normalizer=normalize_audio)

    assert counts(report) == (1, 0, 0, 0)
    row = ledger.get(1)
    stored = audio_path(data_dir, row["audio_sha256"]).read_bytes()
    assert sha(stored) == row["audio_sha256"]
    with wave.open(io.BytesIO(stored), "rb") as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 16000)
    assert row["duration"] == pytest.approx(1.0, abs=0.05)


# --- run_sync and the CLI ---------------------------------------------------


def test_run_sync_opens_the_ledger_under_data_dir_and_reads_env(fake, data_dir, monkeypatch):
    fake.add(1, speaker=42)
    monkeypatch.setenv("AUDITOR_STT_SPEAKER_SALT", "pepper")
    client = CrowdSourceVoiceClient(BASE_URL, "test-token", transport=httpx.MockTransport(fake))

    report = run_sync(BASE_URL, 3, data_dir, client=client, normalizer=passthrough)

    assert counts(report) == (1, 0, 0, 0)
    with Ledger(data_dir / "ledger.sqlite") as ledger:
        assert ledger.get(1)["speaker_id"] == _expected_hash("pepper", 42)
    client.close()


def test_run_sync_requires_the_token_env_var(data_dir, monkeypatch):
    monkeypatch.delenv("CSV_ADMIN_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="CSV_ADMIN_TOKEN"):
        run_sync(BASE_URL, 3, data_dir)


def test_report_summary_is_printable():
    report = SyncReport(added=2, updated=1, removed=0, unchanged=5, skipped={"download_failed": 2, "empty_text": 1})

    assert str(report) == "added 2, updated 1, removed 0, unchanged 5; skipped 3 (download_failed=2, empty_text=1)"
    assert str(SyncReport()) == "added 0, updated 0, removed 0, unchanged 0"


def test_report_summary_names_previously_rejected_recordings():
    report = SyncReport(unchanged=4)
    report.skip("previously_rejected")
    report.skip("previously_rejected")

    assert report.skipped == {"previously_rejected": 2}
    assert str(report) == "added 0, updated 0, removed 0, unchanged 4; skipped 2 (previously_rejected=2)"


def test_cli_dataset_sync_passes_its_arguments_to_run_sync(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run_sync(**kwargs):
        seen.update(kwargs)
        return SyncReport(added=2, unchanged=1)

    monkeypatch.setattr("auditor_stt.dataset.sync.run_sync", fake_run_sync)
    monkeypatch.setenv("AUDITOR_STT_TRAIN_DATA_DIR", str(tmp_path / "from-env"))

    code = main(["dataset", "sync", "--base-url", BASE_URL, "--corpus-id", "3"])

    assert code == 0
    assert seen == {
        "base_url": BASE_URL,
        "corpus_id": 3,
        "data_dir": str(tmp_path / "from-env"),
        "token_env": "CSV_ADMIN_TOKEN",
        "speaker_salt_env": "AUDITOR_STT_SPEAKER_SALT",
        "allow_mass_removal": False,
    }
    assert "added 2, updated 0, removed 0, unchanged 1" in capsys.readouterr().out


def test_cli_dataset_sync_options_override_the_defaults(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("auditor_stt.dataset.sync.run_sync", lambda **kwargs: seen.update(kwargs) or SyncReport())

    code = main([
        "dataset", "sync", "--base-url", BASE_URL, "--corpus-id", "9", "--token-env", "MY_TOKEN",
        "--data-dir", str(tmp_path), "--speaker-salt-env", "MY_SALT", "--allow-mass-removal",
    ])

    assert code == 0
    assert (seen["corpus_id"], seen["token_env"], seen["speaker_salt_env"], seen["data_dir"]) == (
        9, "MY_TOKEN", "MY_SALT", str(tmp_path),
    )
    assert seen["allow_mass_removal"] is True


@pytest.mark.parametrize(
    "error",
    [
        MassRemovalError("Refusing to remove 8 of 10 recordings"),
        ExportInconsistentError("Export and manifest still disagree"),
        UnsupportedCorpusTypeError("Corpus 3 is type 'music'"),
        RuntimeError("Env var CSV_ADMIN_TOKEN is not set"),
        httpx.ConnectError("connection refused"),
    ],
)
def test_cli_dataset_sync_reports_named_errors_with_a_non_zero_exit(monkeypatch, tmp_path, capsys, error):
    def failing_run_sync(**kwargs):
        raise error

    monkeypatch.setattr("auditor_stt.dataset.sync.run_sync", failing_run_sync)

    code = main(["dataset", "sync", "--base-url", BASE_URL, "--corpus-id", "3", "--data-dir", str(tmp_path)])

    assert code != 0
    assert str(error) in capsys.readouterr().err


def test_cli_dataset_sync_requires_base_url_and_corpus_id():
    with pytest.raises(SystemExit) as excinfo:
        main(["dataset", "sync", "--base-url", BASE_URL])

    assert excinfo.value.code == 2


def test_a_conversion_timeout_is_transient_and_not_remembered_as_a_rejection(fake, ledger, data_dir):
    # An unreadable file is rejected for good; a timeout only means the machine was busy.
    fake.add(1, seconds=2.0)
    attempts = []

    def slow_then_fast(data, source_name):
        attempts.append(source_name)
        if len(attempts) == 1:
            raise AudioNormalizeTimeout("ffmpeg timed out")
        return data

    first = sync(fake, ledger, data_dir, normalizer=slow_then_fast, now=T0)

    assert first.skipped == {"audio_timeout": 1}
    assert counts(first) == (0, 0, 0, 0)
    assert ledger.rejected(1) is None
    assert ledger.get(1) is None

    second = sync(fake, ledger, data_dir, normalizer=slow_then_fast, now=T1)

    assert counts(second) == (1, 0, 0, 0)
    assert len(attempts) == 2  # the second sync downloaded and converted it again
    assert ledger.rejected(1) is None
