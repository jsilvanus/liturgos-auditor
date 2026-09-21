import hashlib
import io
import json
import struct
import wave
from pathlib import Path

import pytest

from auditor_stt.cli import main
from auditor_stt.dataset.audio import audio_path, store_audio
from auditor_stt.dataset.build import build_from_ledger, datasets_root
from auditor_stt.dataset.ledger import Ledger, ledger_path
from auditor_stt.dataset.lineage import prune, purge_speaker, speaker_lineage

SPEAKER_A = "speakerA00000000000000"
SPEAKER_B = "speakerB00000000000000"
T1 = "2026-01-01T00:00:00Z"
T2 = "2026-02-01T00:00:00Z"
T3 = "2026-03-01T00:00:00Z"
T4 = "2026-04-01T00:00:00Z"


def _row(recording_id, speaker, split="train"):
    return {
        "recording_id": recording_id, "speaker_id": speaker, "split": split,
        "audio_sha256": "0" * 64, "text_hash": "1" * 64, "duration": 1.0,
    }


def _fake_version(data_dir, version, rows, created_at=T1):
    """A built dataset version as far as lineage/purge/prune can tell: manifest, metadata, some data files."""
    path = datasets_root(data_dir) / version
    (path / "train").mkdir(parents=True)
    (path / "train" / "data-00000.arrow").write_bytes(b"embedded copy of the audio " + version.encode())
    (path / "dataset_dict.json").write_text("{}")
    manifest = {"manifest_version": 1, "dataset_version": version, "rows": rows}
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "build_metadata.json").write_text(json.dumps({"dataset_version": version, "created_at": created_at}))
    return path


def _model(models_dir, name, metadata, filename="training_metadata.json"):
    path = Path(models_dir) / name
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"weights of " + name.encode())
    (path / filename).write_text(json.dumps(metadata))
    return path


def _tree(root):
    root = Path(root)
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _wav(seed):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", seed) * 1600)
    return buf.getvalue()


def _record(data_dir, ledger, rid, speaker, audio_seed, text="salainen lause"):
    wav = _wav(audio_seed)
    sha = hashlib.sha256(wav).hexdigest()
    store_audio(data_dir, sha, wav)
    ledger.upsert_active(
        rid, 3, speaker_id=speaker, text=text, audio_sha256=sha, duration=0.1, quality_score=4.0,
        audio_path=f"uploads/audio/{rid}.wav",
    )
    return sha


@pytest.fixture
def world(tmp_path):
    """Speaker A: recordings 1 (audio X) and 2 (audio Y). Speaker B: 3 (audio Y, shared with A) and 4 (audio Z).

    Dataset `va` holds both speakers, `vb` only B. Model `run_a` was trained on `va`, `run_b` on `vb`,
    and there is an interrupted build.
    """
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    with Ledger(ledger_path(data_dir)) as ledger:
        sha = {
            "X": _record(data_dir, ledger, 1, SPEAKER_A, 1),
            "Y": _record(data_dir, ledger, 2, SPEAKER_A, 2),
            "Z": _record(data_dir, ledger, 4, SPEAKER_B, 3),
        }
        ledger.upsert_active(
            3, 3, speaker_id=SPEAKER_B, text="salainen lause", audio_sha256=sha["Y"], duration=0.1,
            quality_score=4.0, audio_path="uploads/audio/3.wav",
        )
    _fake_version(data_dir, "va", [_row(1, SPEAKER_A, "dev"), _row(2, SPEAKER_A, "dev"), _row(3, SPEAKER_B)], T1)
    _fake_version(data_dir, "vb", [_row(3, SPEAKER_B), _row(4, SPEAKER_B)], T2)
    partial = datasets_root(data_dir) / "interrupted"
    (partial / "train").mkdir(parents=True)
    (partial / "train" / "data-00000.arrow").write_bytes(b"half written copy of the audio")
    _model(models_dir, "run_a", {"dataset_version": "va", "epochs": 3})
    _model(models_dir, "run_b", {"dataset_version": "vb"})
    return {"data_dir": data_dir, "models_dir": models_dir, "sha": sha, "partial": partial}


# --- lineage ------------------------------------------------------------------------


def test_lineage_lists_dataset_versions_and_models_that_used_the_speaker(tmp_path):
    data_dir, models_dir = tmp_path / "data", tmp_path / "models"
    _fake_version(data_dir, "v1", [_row(1, SPEAKER_A), _row(2, SPEAKER_A), _row(3, SPEAKER_B)])
    _fake_version(data_dir, "v2", [_row(3, SPEAKER_B)])
    _fake_version(data_dir, "v3", [_row(1, SPEAKER_A, "test")])
    (datasets_root(data_dir) / "v4").mkdir()
    (datasets_root(data_dir) / "v4" / "SUPERSEDED.json").write_text("{}")
    _model(models_dir, "run1", {"dataset_version": "v1"})
    _model(models_dir / "registry" / "fi", "1.0", {"training": {"dataset_version": "v3"}}, filename="metadata.json")
    _model(models_dir, "other", {"dataset_version": "v2"})
    _model(models_dir, "legacy", {"dataset_version": None})

    report = speaker_lineage(data_dir, SPEAKER_A, models_dir)

    assert [(d["dataset_version"], d["recordings"], d["splits"]) for d in report["datasets"]] == [
        ("v1", 2, ["train"]), ("v3", 1, ["test"]),
    ]
    assert [(Path(m["path"]).name, m["dataset_version"], m["metadata_file"]) for m in report["models"]] == [
        ("1.0", "v3", "metadata.json"), ("run1", "v1", "training_metadata.json"),
    ]
    assert report["models_dir_found"] is True
    assert report["ledger_recordings"] == 0

    unknown = speaker_lineage(data_dir, "nobody", models_dir)
    assert unknown["datasets"] == [] and unknown["models"] == []


def test_lineage_counts_active_ledger_recordings_and_survives_missing_dirs(world, tmp_path):
    report = speaker_lineage(world["data_dir"], SPEAKER_A, tmp_path / "no-models")

    assert report["ledger_recordings"] == 2
    assert [d["dataset_version"] for d in report["datasets"]] == ["va"]
    assert report["models_dir_found"] is False and report["models"] == []

    empty = speaker_lineage(tmp_path / "empty", SPEAKER_A)
    assert empty["ledger_recordings"] == 0 and empty["datasets"] == []
    assert not (tmp_path / "empty").exists()  # a read-only query creates nothing


def test_cli_lineage_text_and_json(world, capsys):
    argv = ["dataset", "lineage", "--data-dir", str(world["data_dir"]), "--models-dir", str(world["models_dir"])]

    assert main([*argv, "--speaker", SPEAKER_A]) == 0
    text = capsys.readouterr().out
    assert "Dataset versions containing this speaker: 1" in text
    assert "va  3 recordings" not in text and "va  2 recordings" in text
    assert "run_a" in text and "run_b" not in text
    assert "salainen" not in text

    assert main([*argv, "--speaker", SPEAKER_A, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["speaker_id"] == SPEAKER_A
    assert report["datasets"][0]["dataset_version"] == "va"
    assert [Path(m["path"]).name for m in report["models"]] == ["run_a"]

    with pytest.raises(SystemExit) as info:
        main(["dataset", "lineage"])
    assert info.value.code == 2


# --- purge --------------------------------------------------------------------------


def _state(world):
    return {
        "datasets": _tree(datasets_root(world["data_dir"])),
        "audio": _tree(world["data_dir"] / "audio"),
        "models": _tree(world["models_dir"]),
    }


def test_purge_dry_run_reports_and_changes_nothing(world, capsys):
    before = _state(world)

    report = purge_speaker(world["data_dir"], SPEAKER_A, world["models_dir"])

    assert _state(world) == before
    with Ledger(ledger_path(world["data_dir"])) as ledger:
        assert ledger.counts() == {"active": 4, "removed": 0}
    assert report["applied"] is False
    assert report["ledger_recordings"] == 2
    assert report["audio_files"] == 1 and report["audio_files_shared"] == 1
    assert report["datasets"] == [{"dataset_version": "va", "recordings": 2}]
    assert report["interrupted_builds"] == 1
    assert [Path(m["path"]).name for m in report["models"]] == ["run_a"]

    argv = ["dataset", "purge", "--data-dir", str(world["data_dir"]), "--models-dir", str(world["models_dir"])]
    assert main([*argv, "--speaker", SPEAKER_A]) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out and "would be tombstoned" in out and "retrain or retire required" in out
    assert SPEAKER_A not in out
    assert _state(world) == before


def test_purge_yes_erases_the_speaker_and_only_reports_models(world, capsys):
    before = _state(world)
    vb_before = _tree(datasets_root(world["data_dir"]) / "vb")
    argv = ["dataset", "purge", "--data-dir", str(world["data_dir"]), "--models-dir", str(world["models_dir"])]

    assert main([*argv, "--speaker", SPEAKER_A, "--yes"]) == 0
    out = capsys.readouterr().out

    with Ledger(ledger_path(world["data_dir"])) as ledger:
        assert ledger.counts() == {"active": 2, "removed": 2}
        for rid in (1, 2):
            row = ledger.get(rid)
            assert row["status"] == "removed" and row["speaker_id"] is None and row["text"] is None
        assert ledger.get(3)["status"] == "active" and ledger.get(4)["speaker_id"] == SPEAKER_B
    # X was only A's; Y is still B's; Z is B's.
    assert not audio_path(world["data_dir"], world["sha"]["X"]).exists()
    assert audio_path(world["data_dir"], world["sha"]["Y"]).exists()
    assert audio_path(world["data_dir"], world["sha"]["Z"]).exists()

    va = datasets_root(world["data_dir"]) / "va"
    assert [p.name for p in va.iterdir()] == ["SUPERSEDED.json"]
    marker_text = (va / "SUPERSEDED.json").read_text()
    marker = json.loads(marker_text)
    assert set(marker) == {"dataset_version", "superseded_at", "reason", "purged_recordings"}
    assert marker["dataset_version"] == "va" and marker["reason"] == "speaker purge" and marker["purged_recordings"] == 2
    assert SPEAKER_A not in marker_text and SPEAKER_B not in marker_text
    assert not world["partial"].exists()
    assert _tree(datasets_root(world["data_dir"]) / "vb") == vb_before

    assert _state(world)["models"] == before["models"]  # models are never touched
    assert "run_a" in out and "run_b" not in out and "retrain or retire required" in out
    assert SPEAKER_A not in out


def test_purge_is_idempotent_and_keeps_reporting_models_of_superseded_versions(world):
    first = purge_speaker(world["data_dir"], SPEAKER_A, world["models_dir"], apply=True)
    settled = _state(world)
    marker = (datasets_root(world["data_dir"]) / "va" / "SUPERSEDED.json").read_bytes()

    second = purge_speaker(world["data_dir"], SPEAKER_A, world["models_dir"], apply=True)

    assert first["ledger_recordings"] == 2
    assert second["ledger_recordings"] == 0 and second["audio_files"] == 0 and second["datasets"] == []
    assert second["interrupted_builds"] == 0
    assert [Path(m["path"]).name for m in second["models"]] == ["run_a"]
    assert _state(world) == settled
    assert (datasets_root(world["data_dir"]) / "va" / "SUPERSEDED.json").read_bytes() == marker


def test_purge_finishes_an_interrupted_run(world):
    # Ledger and audio already erased (crash before the dataset step): the datasets still get superseded.
    with Ledger(ledger_path(world["data_dir"])) as ledger:
        ledger.mark_removed([1, 2])
    audio_path(world["data_dir"], world["sha"]["X"]).unlink()

    report = purge_speaker(world["data_dir"], SPEAKER_A, world["models_dir"], apply=True)

    assert report["ledger_recordings"] == 0
    assert report["datasets"] == [{"dataset_version": "va", "recordings": 2}]
    assert [p.name for p in (datasets_root(world["data_dir"]) / "va").iterdir()] == ["SUPERSEDED.json"]


def test_a_shared_audio_file_goes_once_its_last_active_recording_is_purged(world):
    purge_speaker(world["data_dir"], SPEAKER_B, world["models_dir"], apply=True)
    assert not audio_path(world["data_dir"], world["sha"]["Z"]).exists()
    assert audio_path(world["data_dir"], world["sha"]["Y"]).exists()  # A's recording 2 still uses it

    purge_speaker(world["data_dir"], SPEAKER_A, world["models_dir"], apply=True)
    assert not (world["data_dir"] / "audio" / f"{world['sha']['Y']}.wav").exists()
    assert _tree(world["data_dir"] / "audio") == {}


def test_purge_of_an_unknown_speaker_or_missing_dirs_is_harmless(tmp_path, capsys):
    report = purge_speaker(tmp_path / "nothing", "nobody", tmp_path / "no-models", apply=True)

    assert report["ledger_recordings"] == 0 and report["datasets"] == [] and report["models"] == []
    assert not (tmp_path / "nothing").exists()
    assert main(["dataset", "purge", "--data-dir", str(tmp_path / "nothing"), "--speaker", "nobody", "--yes"]) == 0
    assert "--models-dir not found" in capsys.readouterr().out


def test_purge_removes_a_real_built_dataset_and_lineage_finds_it_first(tmp_path):
    data_dir = tmp_path / "data"
    with Ledger(ledger_path(data_dir)) as ledger:
        for rid in range(1, 61):
            _record(data_dir, ledger, rid, f"spk{rid}", rid, text=f"lause numero {rid}")
    metadata = build_from_ledger(data_dir, split_ratios={"train": 0.6, "dev": 0.2, "test": 0.2})
    manifest = json.loads((datasets_root(data_dir) / metadata["dataset_version"] / "manifest.json").read_text())
    victim = next(row for row in manifest["rows"] if row["split"] == "dev")
    models_dir = tmp_path / "models"
    _model(models_dir, "run", {"dataset_version": metadata["dataset_version"], "dataset": "/somewhere"})

    before = speaker_lineage(data_dir, victim["speaker_id"], models_dir)
    assert [(d["dataset_version"], d["recordings"], d["splits"]) for d in before["datasets"]] == [
        (metadata["dataset_version"], 1, ["dev"])
    ]
    assert len(before["models"]) == 1 and before["ledger_recordings"] == 1

    purge_speaker(data_dir, victim["speaker_id"], models_dir, apply=True)

    version_dir = datasets_root(data_dir) / metadata["dataset_version"]
    assert [p.name for p in version_dir.iterdir()] == ["SUPERSEDED.json"]
    assert not audio_path(data_dir, victim["audio_sha256"]).exists()
    assert speaker_lineage(data_dir, victim["speaker_id"], models_dir)["datasets"] == []
    # Rebuilding what is left works and gets a new version.
    rebuilt = build_from_ledger(data_dir, split_ratios={"train": 0.6, "dev": 0.2, "test": 0.2})
    assert rebuilt["dataset_version"] != metadata["dataset_version"]


# --- prune --------------------------------------------------------------------------


def _four_versions(data_dir):
    # Names sort opposite to age, so only created_at can put them in order.
    for version, created in (("d-oldest", T1), ("c-old", T2), ("b-new", T3), ("a-newest", T4)):
        _fake_version(data_dir, version, [_row(1, SPEAKER_A)], created)


def _versions(data_dir):
    return sorted(p.name for p in datasets_root(data_dir).iterdir())


def test_prune_keeps_the_newest_versions_by_created_at(tmp_path):
    _four_versions(tmp_path)

    preview = prune(tmp_path, keep_datasets=2)
    assert [d["dataset_version"] for d in preview["datasets"]] == ["c-old", "d-oldest"]
    assert preview["datasets_total"] == 4
    assert len(_versions(tmp_path)) == 4  # dry run

    done = prune(tmp_path, keep_datasets=2, apply=True)
    assert [d["dataset_version"] for d in done["datasets"]] == ["c-old", "d-oldest"]
    assert _versions(tmp_path) == ["a-newest", "b-new"]
    assert prune(tmp_path, keep_datasets=2, apply=True)["datasets"] == []


def test_prune_spares_versions_a_model_still_names(tmp_path):
    data_dir, models_dir = tmp_path / "data", tmp_path / "models"
    _four_versions(data_dir)
    _model(models_dir, "run", {"dataset_version": "d-oldest"})

    protected = prune(data_dir, keep_datasets=1, models_dir=models_dir, apply=True)
    assert [d["dataset_version"] for d in protected["datasets"]] == ["b-new", "c-old"]
    assert [d["dataset_version"] for d in protected["datasets_protected"]] == ["d-oldest"]
    assert _versions(data_dir) == ["a-newest", "d-oldest"]

    forced = prune(data_dir, keep_datasets=1, models_dir=models_dir, include_referenced=True, apply=True)
    assert [d["dataset_version"] for d in forced["datasets"]] == ["d-oldest"]
    assert _versions(data_dir) == ["a-newest"]
    assert (models_dir / "run" / "model.safetensors").exists()  # models are never pruned


def test_prune_ignores_superseded_markers_and_interrupted_builds(tmp_path):
    _four_versions(tmp_path)
    (datasets_root(tmp_path) / "gone").mkdir()
    (datasets_root(tmp_path) / "gone" / "SUPERSEDED.json").write_text("{}")
    (datasets_root(tmp_path) / "half" / "train").mkdir(parents=True)

    report = prune(tmp_path, keep_datasets=3, apply=True)

    assert [d["dataset_version"] for d in report["datasets"]] == ["d-oldest"]
    assert _versions(tmp_path) == ["a-newest", "b-new", "c-old", "gone", "half"]


def test_prune_deletes_checkpoints_of_finished_runs_only(tmp_path):
    runs = tmp_path / "runs"
    done = _model(runs, "run1", {"dataset_version": "v1"})
    for step in (100, 200):
        (done / f"checkpoint-{step}").mkdir()
        (done / f"checkpoint-{step}" / "optimizer.pt").write_bytes(b"x" * 1000)
    (done / "checkpoint-best").mkdir()  # not a Trainer checkpoint name
    nested = _model(runs / "experiments", "run2", {"dataset_version": "v1"})
    (nested / "checkpoint-7").mkdir()
    (nested / "checkpoint-7" / "optimizer.pt").write_bytes(b"x" * 500)
    unfinished = runs / "run3"
    (unfinished / "checkpoint-50").mkdir(parents=True)
    (unfinished / "checkpoint-50" / "optimizer.pt").write_bytes(b"x")
    final_files = _tree(done) | _tree(nested)

    preview = prune(tmp_path, runs_dir=runs)
    assert sorted(Path(c["path"]).name for c in preview["checkpoints"]) == ["checkpoint-100", "checkpoint-200", "checkpoint-7"]
    assert sum(c["bytes"] for c in preview["checkpoints"]) == 2500
    assert preview["checkpoints_in_unfinished_runs"] == 1
    assert (done / "checkpoint-100").exists()

    prune(tmp_path, runs_dir=runs, apply=True)

    assert not (done / "checkpoint-100").exists() and not (done / "checkpoint-200").exists()
    assert not (nested / "checkpoint-7").exists()
    assert (done / "checkpoint-best").exists()
    assert (unfinished / "checkpoint-50" / "optimizer.pt").exists()
    remaining = _tree(done) | _tree(nested)
    assert remaining == {k: v for k, v in final_files.items() if "checkpoint-100" not in k and "checkpoint-200" not in k
                         and "checkpoint-7" not in k}
    assert (done / "model.safetensors").exists() and (done / "training_metadata.json").exists()


def test_cli_prune_requires_something_to_prune_and_is_a_dry_run_by_default(tmp_path, capsys):
    data_dir = tmp_path / "data"
    _four_versions(data_dir)

    assert main(["dataset", "prune", "--data-dir", str(data_dir)]) == 2
    assert main(["dataset", "prune", "--data-dir", str(data_dir), "--keep-datasets", "-1"]) == 2
    assert "error" in capsys.readouterr().err

    argv = ["dataset", "prune", "--data-dir", str(data_dir), "--models-dir", str(tmp_path / "models"), "--keep-datasets", "1"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out and "d-oldest" in out and "--models-dir not found" in out
    assert len(_versions(data_dir)) == 4

    assert main([*argv, "--yes"]) == 0
    assert "Dry run" not in capsys.readouterr().out
    assert _versions(data_dir) == ["a-newest"]


def test_cli_prune_checkpoints(tmp_path, capsys):
    run = _model(tmp_path / "runs", "run1", {"dataset_version": "v1"})
    (run / "checkpoint-9").mkdir()
    (run / "checkpoint-9" / "optimizer.pt").write_bytes(b"x")

    assert main(["dataset", "prune", "--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs"), "--yes"]) == 0

    assert "Checkpoint directories deleted: 1" in capsys.readouterr().out
    assert not (run / "checkpoint-9").exists()
    assert (run / "model.safetensors").exists()
