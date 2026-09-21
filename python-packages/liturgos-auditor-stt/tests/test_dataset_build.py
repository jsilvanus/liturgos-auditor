import hashlib
import io
import json
import shutil
import struct
import wave

import pytest

from auditor_stt.cli import main
from auditor_stt.dataset.audio import audio_path, store_audio
from auditor_stt.dataset.build import (
    DEFAULT_SPLIT_RATIOS,
    build_dataset,
    build_from_ledger,
    compute_split,
    datasets_root,
    speaker_bucket,
    speaker_split,
    split_salt,
)
from auditor_stt.dataset.ledger import Ledger, ledger_path
from auditor_stt.dataset.normalize import normalize_text

# Wide dev/test shares so a few dozen speakers fill every split (the default 5% needs hundreds).
RATIOS = {"train": 0.6, "dev": 0.2, "test": 0.2}


def _rows(n, speakers_per=None):
    rows = []
    for i in range(n):
        speaker = f"spk{i % speakers_per}" if speakers_per else None
        rows.append({"file": f"{i:04d}.wav", "speaker_id": speaker, "text": "x", "duration": 1.0})
    return rows


def _speakers_in(split, count, taken=(), ratios=RATIOS, seed=42):
    """`count` speaker ids that the stable hash puts in `split`."""
    salt = split_salt(seed)
    found = []
    for i in range(100_000):
        name = f"spk{i}"
        if name not in taken and speaker_split(name, salt, ratios) == split:
            found.append(name)
            if len(found) == count:
                return found
    raise AssertionError("no such speakers")


# --- compute_split -----------------------------------------------------------------


def test_speaker_disjoint_when_all_rows_have_speaker_id():
    rows = _rows(30, speakers_per=6)
    assignment, speaker_disjoint = compute_split(rows, seed=1)

    assert speaker_disjoint is True
    assert set(assignment.keys()) == {r["file"] for r in rows}

    speaker_of = {r["file"]: r["speaker_id"] for r in rows}
    splits_by_speaker = {}
    for file, split_name in assignment.items():
        splits_by_speaker.setdefault(speaker_of[file], set()).add(split_name)

    assert all(len(splits) == 1 for splits in splits_by_speaker.values())


def test_rows_without_speaker_id_are_train_only():
    # Used to downgrade the whole split to a random utterance-level one.
    rows = _rows(20, speakers_per=None)
    assignment, speaker_disjoint = compute_split(rows, seed=1)

    assert speaker_disjoint is True
    assert set(assignment.keys()) == {r["file"] for r in rows}
    assert set(assignment.values()) == {"train"}


def test_a_row_without_speaker_id_leaves_every_other_row_where_it_was():
    # Used to flip the whole split to the random fallback.
    rows = _rows(60, speakers_per=20)
    before, _ = compute_split(rows, seed=1)
    rows[0]["speaker_id"] = None
    after, speaker_disjoint = compute_split(rows, seed=1)

    assert speaker_disjoint is True
    assert after[rows[0]["file"]] == "train"
    others = [r["file"] for r in rows[1:]]
    assert {f: after[f] for f in others} == {f: before[f] for f in others}


def test_split_is_deterministic_given_seed():
    rows = _rows(20, speakers_per=6)
    a1, _ = compute_split(rows, seed=7)
    a2, _ = compute_split(rows, seed=7)
    assert a1 == a2


def test_split_sizes_roughly_match_ratios():
    rows = _rows(200, speakers_per=50)
    assignment, _ = compute_split(rows, seed=3)
    counts = {"train": 0, "dev": 0, "test": 0}
    for split_name in assignment.values():
        counts[split_name] += 1
    assert counts["train"] > counts["dev"]
    assert counts["train"] > counts["test"]
    assert sum(counts.values()) == 200


def test_speaker_bucket_is_the_documented_hash():
    expected = int(hashlib.sha256(b"auditor-stt-split-v1:42:spkhash1").hexdigest()[:8], 16) % 10000
    assert split_salt(42) == "auditor-stt-split-v1:42"
    assert speaker_bucket("spkhash1", split_salt(42)) == expected


def test_a_speakers_split_depends_only_on_speaker_salt_and_ratios():
    rows = _rows(400, speakers_per=200)
    full, _ = compute_split(rows, seed=5)
    subset, _ = compute_split(rows[::7], seed=5)

    assert all(subset[f] == full[f] for f in subset)
    for row in rows:
        assert full[row["file"]] == speaker_split(row["speaker_id"], split_salt(5), DEFAULT_SPLIT_RATIOS)


def test_seed_is_part_of_the_salt_so_it_changes_the_split():
    rows = _rows(400, speakers_per=200)
    a, _ = compute_split(rows, seed=1)
    b, _ = compute_split(rows, seed=2)
    assert a != b


def test_thresholds_carve_dev_then_test_from_the_low_end():
    salt = split_salt(42)
    ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}
    for i in range(300):
        bucket = speaker_bucket(f"s{i}", salt)
        expected = "dev" if bucket < 1000 else "test" if bucket < 2000 else "train"
        assert speaker_split(f"s{i}", salt, ratios) == expected


def test_ratios_must_name_exactly_train_dev_test():
    with pytest.raises(ValueError, match="train"):
        compute_split(_rows(4, 2), split_ratios={"train": 0.5, "dev": 0.5})
    with pytest.raises(ValueError, match="negative"):
        compute_split(_rows(4, 2), split_ratios={"train": 1.1, "dev": -0.1, "test": 0.0})


def test_normalize_text_strips_prompt_artifacts_and_collapses_whitespace():
    assert normalize_text("  Moi   maailma\\n  ") == "Moi maailma"
    assert normalize_text("rivi1\r\nrivi2") == "rivi1 rivi2"


def test_normalize_text_preserves_finnish_orthography_and_case():
    text = "Hyvää huomenta, Räätälintie 3!"
    assert normalize_text(text) == text


def test_normalize_text_passes_through_none():
    assert normalize_text(None) is None


# --- build_from_ledger --------------------------------------------------------------


def _wav(seed):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", seed) * 1600)
    return buf.getvalue()


def _add(data_dir, ledger, rid, speaker, text=None, corpus_id=3, audio_seed=None):
    wav = _wav(rid if audio_seed is None else audio_seed)
    sha = hashlib.sha256(wav).hexdigest()
    store_audio(data_dir, sha, wav)
    ledger.upsert_active(
        rid, corpus_id, speaker_id=speaker, text=text or f"lause numero {rid}", audio_sha256=sha, duration=0.1,
        quality_score=4.0, audio_path=f"uploads/audio/{rid}.wav",
    )
    return sha


def _speaker_map(dataset_dir):
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return {row["speaker_id"]: row["split"] for row in manifest["rows"] if row["speaker_id"]}


def _out_dir(data_dir, metadata):
    return datasets_root(data_dir) / metadata["dataset_version"]


@pytest.fixture
def grown_corpus(tmp_path):
    """40 speakers x 2 recordings; `grow()` adds 40 new speakers, more recordings for old ones, and anonymous rows."""
    counter = iter(range(1, 10_000))
    with Ledger(ledger_path(tmp_path)) as ledger:
        for i in range(40):
            for _ in range(2):
                _add(tmp_path, ledger, next(counter), f"spk{i}")

    def grow():
        with Ledger(ledger_path(tmp_path)) as ledger:
            for i in range(40):
                _add(tmp_path, ledger, next(counter), f"new{i}")
            for i in range(0, 40, 3):
                _add(tmp_path, ledger, next(counter), f"spk{i}")

    return tmp_path, grow


def test_existing_speakers_keep_their_split_when_the_corpus_grows(grown_corpus):
    data_dir, grow = grown_corpus
    v1 = build_from_ledger(data_dir, split_ratios=RATIOS)
    v1_speakers = _speaker_map(_out_dir(data_dir, v1))
    assert set(v1_speakers.values()) == {"train", "dev", "test"}

    grow()
    v2 = build_from_ledger(data_dir, split_ratios=RATIOS)
    v2_speakers = _speaker_map(_out_dir(data_dir, v2))

    assert v2["dataset_version"] != v1["dataset_version"]
    assert len(v2_speakers) == 80
    assert all(v2_speakers[speaker] == split for speaker, split in v1_speakers.items())
    assert v2["speaker_disjoint"] is True
    assert v2["split_salt"] == v1["split_salt"] == "auditor-stt-split-v1:42"


def test_rows_without_speaker_are_train_only_and_do_not_move_anyone(grown_corpus):
    data_dir, _ = grown_corpus
    v1 = build_from_ledger(data_dir, split_ratios=RATIOS)
    before = _speaker_map(_out_dir(data_dir, v1))

    with Ledger(ledger_path(data_dir)) as ledger:
        for rid in range(500, 510):
            _add(data_dir, ledger, rid, None)
    v2 = build_from_ledger(data_dir, split_ratios=RATIOS)

    assert _speaker_map(_out_dir(data_dir, v2)) == before
    assert v1["train_rows_without_speaker"] == 0
    assert v2["train_rows_without_speaker"] == 10
    assert v2["split_sizes"]["train"] == v1["split_sizes"]["train"] + 10
    assert v2["split_sizes"]["dev"] == v1["split_sizes"]["dev"]
    assert v2["split_sizes"]["test"] == v1["split_sizes"]["test"]
    assert v2["speaker_counts"] == v1["speaker_counts"]
    assert v2["speaker_disjoint"] is True


def test_transcript_leakage_guard_drops_train_rows_that_repeat_an_eval_sentence(tmp_path):
    train_a, train_b = _speakers_in("train", 2)
    dev_speaker, = _speakers_in("dev", 1)
    test_speaker, = _speakers_in("test", 1)
    with Ledger(ledger_path(tmp_path)) as ledger:
        _add(tmp_path, ledger, 1, dev_speaker, "Hyvää huomenta, seurakunta.")
        _add(tmp_path, ledger, 2, test_speaker, "Herra on minun paimeneni.")
        # Read by train speakers, and once by an anonymous recorder: all three must go.
        _add(tmp_path, ledger, 3, train_a, "Hyvää huomenta, seurakunta.")
        _add(tmp_path, ledger, 4, train_b, "Herra on minun paimeneni.")
        _add(tmp_path, ledger, 5, None, "Hyvää huomenta, seurakunta.")
        # Kept: a sentence no eval speaker read.
        _add(tmp_path, ledger, 6, train_a, "Ylistäkää Herraa.")
        _add(tmp_path, ledger, 7, train_b, "Ylistäkää Herraa.")

    metadata = build_from_ledger(tmp_path, split_ratios=RATIOS)

    assert metadata["train_rows_dropped_for_transcript_overlap"] == 3
    assert metadata["split_sizes"] == {"train": 2, "dev": 1, "test": 1}
    manifest = json.loads((_out_dir(tmp_path, metadata) / "manifest.json").read_text(encoding="utf-8"))
    by_split = {}
    for row in manifest["rows"]:
        by_split.setdefault(row["split"], set()).add(row["recording_id"])
    assert by_split == {"train": {6, 7}, "dev": {1}, "test": {2}}
    train_hashes = {r["text_hash"] for r in manifest["rows"] if r["split"] == "train"}
    eval_hashes = {r["text_hash"] for r in manifest["rows"] if r["split"] != "train"}
    assert not train_hashes & eval_hashes


def test_leakage_guard_compares_normalised_text(tmp_path):
    train_speaker, = _speakers_in("train", 1)
    dev_speaker, = _speakers_in("dev", 1)
    test_speaker, = _speakers_in("test", 1)
    with Ledger(ledger_path(tmp_path)) as ledger:
        _add(tmp_path, ledger, 1, dev_speaker, "Moi maailma")
        _add(tmp_path, ledger, 2, test_speaker, "Toinen lause")
        _add(tmp_path, ledger, 3, train_speaker, "  Moi   maailma \\n")
        _add(tmp_path, ledger, 4, train_speaker, "Kolmas lause")

    metadata = build_from_ledger(tmp_path, split_ratios=RATIOS)

    assert metadata["train_rows_dropped_for_transcript_overlap"] == 1
    assert metadata["split_sizes"]["train"] == 1


def test_identical_inputs_give_the_same_version_and_rebuilding_is_a_noop(grown_corpus):
    data_dir, _ = grown_corpus
    first = build_from_ledger(data_dir, split_ratios=RATIOS)
    out_dir = _out_dir(data_dir, first)
    (out_dir / "sentinel.txt").write_text("still here")

    again = build_from_ledger(data_dir, split_ratios=RATIOS)
    assert again == first
    assert (out_dir / "sentinel.txt").exists()

    forced = build_from_ledger(data_dir, split_ratios=RATIOS, force=True)
    assert forced["dataset_version"] == first["dataset_version"]
    assert forced["manifest_sha256"] == first["manifest_sha256"]
    assert not (out_dir / "sentinel.txt").exists()
    assert (out_dir / "build_metadata.json").is_file()


def test_the_version_is_a_pure_function_of_the_inputs(tmp_path):
    def populate(data_dir, text_for_last="lause"):
        with Ledger(ledger_path(data_dir)) as ledger:
            for i in range(60):  # enough speakers that every split stays non-empty under the other seed and ratios below
                _add(data_dir, ledger, i + 1, f"spk{i}", text_for_last if i == 59 else None)

    populate(tmp_path / "a")
    populate(tmp_path / "b")
    a = build_from_ledger(tmp_path / "a", split_ratios=RATIOS)
    b = build_from_ledger(tmp_path / "b", split_ratios=RATIOS)
    assert a["dataset_version"] == b["dataset_version"]
    assert a["manifest_sha256"] == b["manifest_sha256"]
    assert len(a["dataset_version"]) == 16

    populate(tmp_path / "c", text_for_last="toinen lause")
    assert build_from_ledger(tmp_path / "c", split_ratios=RATIOS)["dataset_version"] != a["dataset_version"]
    assert build_from_ledger(tmp_path / "a", split_ratios=RATIOS, seed=7)["dataset_version"] != a["dataset_version"]
    other_ratios = {"train": 0.5, "dev": 0.25, "test": 0.25}
    assert build_from_ledger(tmp_path / "a", split_ratios=other_ratios)["dataset_version"] != a["dataset_version"]


def test_manifest_lists_every_row_without_its_text(tmp_path):
    speakers = _speakers_in("train", 3) + _speakers_in("dev", 2) + _speakers_in("test", 2)
    texts = [f"salainen lause {i}" for i in range(len(speakers))]
    with Ledger(ledger_path(tmp_path)) as ledger:
        shas = [_add(tmp_path, ledger, i + 1, speaker, texts[i]) for i, speaker in enumerate(speakers)]

    metadata = build_from_ledger(tmp_path, split_ratios=RATIOS)
    out_dir = _out_dir(tmp_path, metadata)
    raw = (out_dir / "manifest.json").read_bytes()
    manifest = json.loads(raw)

    assert metadata["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["dataset_version"] == metadata["dataset_version"]
    assert [row["recording_id"] for row in manifest["rows"]] == list(range(1, 8))
    for row in manifest["rows"]:
        assert set(row) == {"recording_id", "speaker_id", "split", "audio_sha256", "text_hash", "duration"}
    assert manifest["rows"][0]["audio_sha256"] == shas[0]
    assert manifest["rows"][0]["speaker_id"] == speakers[0]
    assert manifest["rows"][0]["duration"] == pytest.approx(0.1)
    assert not any(text.encode() in raw for text in texts)
    assert b'"text"' not in raw


def test_build_metadata_keys_and_counts(grown_corpus):
    data_dir, _ = grown_corpus
    metadata = build_from_ledger(data_dir, corpus_id=3, seed=9, split_ratios=RATIOS)

    on_disk = json.loads((_out_dir(data_dir, metadata) / "build_metadata.json").read_text(encoding="utf-8"))
    assert on_disk == metadata
    assert set(metadata) >= {
        "dataset_version", "manifest_sha256", "created_at", "source", "seed", "split_ratios", "split_salt",
        "split_sizes", "speaker_counts", "train_rows_without_speaker", "train_rows_dropped_for_transcript_overlap",
        "speaker_disjoint",
    }
    assert metadata["source"] == {"kind": "ledger", "corpus_ids": [3]}
    assert metadata["seed"] == 9
    assert metadata["split_salt"] == "auditor-stt-split-v1:9"
    assert metadata["split_ratios"] == RATIOS
    assert sum(metadata["split_sizes"].values()) == 80
    assert sum(metadata["speaker_counts"].values()) == 40
    assert metadata["created_at"].endswith("Z")


def test_dataset_columns_text_normalisation_and_embedded_audio(tmp_path):
    from datasets import Audio, load_from_disk

    speakers = _speakers_in("train", 3) + _speakers_in("dev", 1) + _speakers_in("test", 1)
    with Ledger(ledger_path(tmp_path)) as ledger:
        for i, speaker in enumerate(speakers):
            _add(tmp_path, ledger, i + 1, speaker, f"  Rivi   {i}\n")
        _add(tmp_path, ledger, 99, None, "Nimetön")

    metadata = build_from_ledger(tmp_path, split_ratios=RATIOS)
    dataset = load_from_disk(str(_out_dir(tmp_path, metadata)))

    assert set(dataset) == {"train", "dev", "test"}
    expected = {"audio", "text", "duration", "speaker_id", "recording_id", "quality_score"}
    assert all(set(split.column_names) == expected for split in dataset.values())
    assert dataset["train"].features["audio"].sampling_rate == 16000
    train = dataset["train"].remove_columns("audio").to_list()
    assert "Rivi 0" in {row["text"] for row in train}
    assert {row["recording_id"] for row in train} == {1, 2, 3, 99}
    assert [row["speaker_id"] for row in train if row["recording_id"] == 99] == [None]

    # The dataset carries its own copy of the audio: it still reads after the cache is gone.
    shutil.rmtree(tmp_path / "audio")
    raw = dataset["train"].cast_column("audio", Audio(decode=False))[0]["audio"]
    assert raw["bytes"] and raw["bytes"][:4] == b"RIFF"


def test_only_active_rows_of_the_requested_corpus_are_used(tmp_path):
    train, dev, test = _speakers_in("train", 5), _speakers_in("dev", 2), _speakers_in("test", 2)
    corpus_3 = train[:3] + dev[:1] + test[:1]
    corpus_4 = train[3:4] + dev[1:] + test[1:]
    with Ledger(ledger_path(tmp_path)) as ledger:
        for i, speaker in enumerate(corpus_3):
            _add(tmp_path, ledger, i + 1, speaker, corpus_id=3)
        for i, speaker in enumerate(corpus_4):
            _add(tmp_path, ledger, 50 + i, speaker, corpus_id=4)
        _add(tmp_path, ledger, 60, train[4], corpus_id=3)
        ledger.mark_removed([60])

    everything = build_from_ledger(tmp_path, split_ratios=RATIOS)
    only_3 = build_from_ledger(tmp_path, corpus_id=3, split_ratios=RATIOS)

    assert everything["source"]["corpus_ids"] == [3, 4]
    assert sum(everything["split_sizes"].values()) == 8
    assert only_3["source"]["corpus_ids"] == [3]
    assert sum(only_3["split_sizes"].values()) == 5
    manifest = json.loads((_out_dir(tmp_path, only_3) / "manifest.json").read_text(encoding="utf-8"))
    assert {row["recording_id"] for row in manifest["rows"]} == {1, 2, 3, 4, 5}


def test_build_fails_with_an_actionable_error_when_an_active_row_has_no_audio_file(tmp_path):
    speakers = _speakers_in("train", 2) + _speakers_in("dev", 1) + _speakers_in("test", 1)
    with Ledger(ledger_path(tmp_path)) as ledger:
        shas = [_add(tmp_path, ledger, i + 1, speaker) for i, speaker in enumerate(speakers)]
    audio_path(tmp_path, shas[0]).unlink()

    with pytest.raises(FileNotFoundError, match="dataset sync") as info:
        build_from_ledger(tmp_path, split_ratios=RATIOS)
    assert "[1]" in str(info.value)
    assert not datasets_root(tmp_path).exists()


def test_build_without_a_ledger_says_to_sync(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset sync"):
        build_from_ledger(tmp_path)
    assert not ledger_path(tmp_path).exists()


def test_ledger_build_raises_for_empty_dev_and_test(tmp_path):
    # csv anonymises some recordings; with only those there is nobody to evaluate on.
    with Ledger(ledger_path(tmp_path)) as ledger:
        for rid in range(1, 6):
            _add(tmp_path, ledger, rid, None)

    with pytest.raises(ValueError, match="would have 0 rows") as info:
        build_from_ledger(tmp_path)
    assert "without a speaker id" in str(info.value)
    assert not datasets_root(tmp_path).exists()


def test_out_root_overrides_the_default_location(grown_corpus):
    data_dir, _ = grown_corpus
    metadata = build_from_ledger(data_dir, out_root=data_dir / "elsewhere", split_ratios=RATIOS)

    assert (data_dir / "elsewhere" / metadata["dataset_version"] / "build_metadata.json").is_file()
    assert not datasets_root(data_dir).exists()


# --- build_dataset (legacy snapshot) -------------------------------------------------


def _write_snapshot(tmp_path, rows):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for row in rows:
        (audio_dir / row["file"]).write_bytes(b"RIFF0000WAVEfmt " + row["file"].encode())
    (tmp_path / "recordings.json").write_text(json.dumps(rows))


def test_build_dataset_raises_clear_error_instead_of_crashing_on_empty_split(tmp_path):
    # Few distinct speakers under the default 90/5/5 ratios leave dev/test with
    # 0 rows — this used to reach datasets' save_to_disk() and crash with an
    # opaque ZeroDivisionError; it must fail fast with an actionable message.
    rows = [
        {"file": f"{i:04d}.wav", "speaker_id": f"spk{i % 3}", "text": "x", "duration": 1.0}
        for i in range(6)
    ]
    _write_snapshot(tmp_path, rows)

    with pytest.raises(ValueError, match="would have 0 rows"):
        build_dataset(str(tmp_path), str(tmp_path / "out"), seed=1)
    assert not (tmp_path / "out").exists()


def test_build_dataset_succeeds_and_writes_metadata_for_a_healthy_split(tmp_path):
    # 200 speakers: under a stable hash split the default 5% dev/test shares need that many to be non-empty.
    rows = [
        {"file": f"{i:04d}.wav", "speaker_id": f"spk{i % 200}", "text": f"row {i}", "duration": 1.0}
        for i in range(400)
    ]
    _write_snapshot(tmp_path, rows)

    out_dir = tmp_path / "out"
    metadata = build_dataset(str(tmp_path), out_dir=str(out_dir), seed=1)

    assert metadata["speaker_disjoint"] is True
    assert all(count > 0 for count in metadata["split_sizes"].values())
    assert sum(metadata["split_sizes"].values()) == 400
    assert json.loads((out_dir / "build_metadata.json").read_text()) == metadata
    assert metadata["source"] == {"kind": "snapshot", "corpus_ids": [], "snapshot_content_hash": None}


def test_build_dataset_uses_file_names_and_content_hashes_as_ids(tmp_path):
    speakers = _speakers_in("train", 3) + _speakers_in("dev", 1) + _speakers_in("test", 1)
    rows = [
        {"file": f"{i + 1:04d}.wav", "speaker_id": speaker, "text": f"row {i}", "duration": 1.0, "quality_score": 4.5}
        for i, speaker in enumerate(speakers)
    ]
    rows.append({"file": "0099.wav", "speaker_id": None, "text": "anonymous", "duration": 1.0})
    _write_snapshot(tmp_path, rows)
    (tmp_path / "snapshot.json").write_text(json.dumps({"corpus_id": 8, "content_hash": "abc123"}))

    metadata = build_dataset(tmp_path, tmp_path / "out", split_ratios=RATIOS)

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    by_id = {row["recording_id"]: row for row in manifest["rows"]}
    assert set(by_id) == {"0001.wav", "0002.wav", "0003.wav", "0004.wav", "0005.wav", "0099.wav"}
    assert by_id["0001.wav"]["audio_sha256"] == hashlib.sha256(b"RIFF0000WAVEfmt 0001.wav").hexdigest()
    assert by_id["0099.wav"]["split"] == "train"
    assert by_id["0099.wav"]["speaker_id"] is None
    assert metadata["source"] == {"kind": "snapshot", "corpus_ids": [8], "snapshot_content_hash": "abc123"}
    assert metadata["train_rows_without_speaker"] == 1


def test_build_dataset_snapshot_without_speaker_ids_is_refused(tmp_path):
    # A snapshot pulled before csv exported speaker_id: everything is train-only, so nothing to evaluate on.
    rows = [{"file": f"{i:04d}.wav", "speaker_id": None, "text": f"row {i}", "duration": 1.0} for i in range(10)]
    _write_snapshot(tmp_path, rows)

    with pytest.raises(ValueError, match="would have 0 rows"):
        build_dataset(tmp_path, tmp_path / "out")


def _snapshot_of(tmp_path, n_speakers):
    rows = [{"file": f"{i:04d}.wav", "speaker_id": f"s{i}", "text": f"row {i}", "duration": 1.0} for i in range(n_speakers)]
    _write_snapshot(tmp_path, rows)


def test_build_dataset_is_idempotent_and_replaces_a_different_earlier_build(tmp_path):
    ratios = {"train": 0.5, "dev": 0.25, "test": 0.25}
    _snapshot_of(tmp_path, 40)
    out_dir = tmp_path / "out"
    first = build_dataset(tmp_path, out_dir, split_ratios=ratios)
    (out_dir / "sentinel.txt").write_text("still here")

    assert build_dataset(tmp_path, out_dir, split_ratios=ratios) == first
    assert (out_dir / "sentinel.txt").exists()

    changed = build_dataset(tmp_path, out_dir, seed=2, split_ratios=ratios)
    assert changed["dataset_version"] != first["dataset_version"]
    assert not (out_dir / "sentinel.txt").exists()

    forced = build_dataset(tmp_path, out_dir, seed=2, split_ratios=ratios, force=True)
    assert forced["dataset_version"] == changed["dataset_version"]


def test_build_dataset_never_overwrites_an_unrelated_directory(tmp_path):
    ratios = {"train": 0.5, "dev": 0.25, "test": 0.25}
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _snapshot_of(snapshot, 40)
    unrelated = tmp_path / "documents"
    unrelated.mkdir()
    (unrelated / "thesis.docx").write_text("precious")

    with pytest.raises(ValueError, match="does not look like a built dataset"):
        build_dataset(snapshot, unrelated, split_ratios=ratios)
    assert (unrelated / "thesis.docx").read_text() == "precious"


# --- CLI ------------------------------------------------------------------------------


def test_cli_build_from_the_ledger_reports_counts_only(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AUDITOR_STT_TRAIN_DATA_DIR", str(tmp_path))
    with Ledger(ledger_path(tmp_path)) as ledger:
        for rid in range(1, 301):  # the default 5% dev/test shares need this many speakers
            _add(tmp_path, ledger, rid, f"spk{rid}")

    assert main(["dataset", "build", "--corpus-id", "3", "--seed", "42"]) == 0
    out = capsys.readouterr().out
    version = next(p.name for p in datasets_root(tmp_path).iterdir())
    assert f"Dataset {version}" in out
    assert "lause numero" not in out

    assert main(["dataset", "build", "--data-dir", str(tmp_path), "--corpus-id", "3"]) == 0
    assert len(list(datasets_root(tmp_path).iterdir())) == 1  # same inputs, same version


def test_cli_build_snapshot_mode_and_argument_errors(tmp_path, capsys):
    _snapshot_of(tmp_path, 400)

    assert main(["dataset", "build", "--snapshot", str(tmp_path), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "build_metadata.json").is_file()
    assert "train " in capsys.readouterr().out

    assert main(["dataset", "build", "--snapshot", str(tmp_path)]) == 2
    assert main(["dataset", "build", "--snapshot", str(tmp_path), "--out", "x", "--corpus-id", "1"]) == 2
    assert main(["dataset", "build", "--snapshot", str(tmp_path), "--out", "x", "--data-dir", "d"]) == 2
    capsys.readouterr()


def test_cli_build_reports_a_missing_ledger_and_empty_splits_as_errors(tmp_path, capsys):
    assert main(["dataset", "build", "--data-dir", str(tmp_path / "nowhere")]) == 1
    assert "dataset sync" in capsys.readouterr().err

    with Ledger(ledger_path(tmp_path)) as ledger:
        for rid in range(1, 4):
            _add(tmp_path, ledger, rid, None)
    assert main(["dataset", "build", "--data-dir", str(tmp_path)]) == 1
    assert "would have 0 rows" in capsys.readouterr().err
