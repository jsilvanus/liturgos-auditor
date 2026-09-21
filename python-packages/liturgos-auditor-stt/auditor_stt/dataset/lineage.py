"""Erasure, lineage and retention for built datasets and the models trained on them.

A dataset build embeds copies of the audio and a model may have memorised its
training data, so withdrawing a speaker's consent reaches further than the
ledger. `lineage` answers "which dataset versions and models used this
speaker?", `purge` erases the speaker from the ledger, the audio cache and the
dataset versions, and `prune` applies retention (old dataset versions,
intermediate training checkpoints).

Models are never deleted or modified here: whether to retrain or retire one is
an owner/DPO decision, so `purge` only lists them. A purged dataset version is
reduced to a small `SUPERSEDED.json` marker that does not name the speaker.
`purge` also does not stop the next `dataset sync` from restoring recordings
that crowd-source-voice still lists; the upstream deletion has to come first.

Every command previews by default and changes something only with `apply`.
Output carries counts and ids only, never transcripts.
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path

from .audio import audio_path
from .build import (
    BUILD_METADATA_FILENAME,
    MANIFEST_FILENAME,
    SUPERSEDED_FILENAME,
    datasets_root,
)
from .ledger import Ledger, ledger_path, utc_iso

logger = logging.getLogger(__name__)

# Training metadata that names the dataset version a model or run was trained on.
MODEL_METADATA_FILENAMES = ("training_metadata.json", "metadata.json")
# Written last by `train`, so a run without it has no final model yet.
RUN_DONE_FILENAME = "training_metadata.json"
_CHECKPOINT = re.compile(r"checkpoint-\d+")


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tree_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _dataset_dirs(data_dir):
    """(path, state) for each directory under `<data_dir>/datasets`.

    state is 'built' (has build_metadata.json, which is written last),
    'superseded' (only the purge marker is left) or 'partial' (an interrupted build).
    """
    root = datasets_root(data_dir)
    if not root.is_dir():
        return []
    found = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if (path / BUILD_METADATA_FILENAME).is_file():
            found.append((path, "built"))
        elif (path / SUPERSEDED_FILENAME).is_file():
            found.append((path, "superseded"))
        else:
            found.append((path, "partial"))
    return found


def _superseded_versions(data_dir):
    versions = set()
    for path, state in _dataset_dirs(data_dir):
        if state == "superseded":
            marker = _load_json(path / SUPERSEDED_FILENAME)
            versions.add(marker.get("dataset_version", path.name) if isinstance(marker, dict) else path.name)
    return versions


def _versions_named_in(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "dataset_version" and isinstance(value, str):
                yield value
            else:
                yield from _versions_named_in(value)
    elif isinstance(node, list):
        for item in node:
            yield from _versions_named_in(item)


def model_references(models_dir):
    """Every (model or run directory, dataset_version) pair named by a metadata file under `models_dir`.

    Looks recursively at `training_metadata.json` and `metadata.json` (a
    registry copy may nest the version, so the key is searched at any depth).
    """
    if models_dir is None or not Path(models_dir).is_dir():
        return []
    found = {}
    for name in MODEL_METADATA_FILENAMES:
        for path in Path(models_dir).rglob(name):
            content = _load_json(path)
            if content is None:
                logger.warning("Ignoring unreadable %s", path)
                continue
            for version in _versions_named_in(content):
                found[(str(path.parent), version)] = {
                    "path": str(path.parent), "dataset_version": version, "metadata_file": name,
                }
    return [found[key] for key in sorted(found)]


def _speaker_datasets(data_dir, speaker_id):
    """The built dataset versions whose manifest lists this speaker, with the number of their recordings."""
    found = []
    for path, state in _dataset_dirs(data_dir):
        if state != "built":
            continue
        manifest = _load_json(path / MANIFEST_FILENAME)
        if not isinstance(manifest, dict):
            continue
        rows = [r for r in manifest.get("rows", []) if r.get("speaker_id") == speaker_id]
        if rows:
            found.append({
                "dataset_version": manifest.get("dataset_version", path.name),
                "path": str(path),
                "recordings": len(rows),
                "splits": sorted({r["split"] for r in rows}),
            })
    return found


def _speaker_ledger_rows(ledger, speaker_id):
    return [
        {"recording_id": row["recording_id"], "audio_sha256": row["audio_sha256"]}
        for row in ledger.active_rows()
        if row["speaker_id"] == speaker_id
    ]


def speaker_lineage(data_dir, speaker_id, models_dir=None):
    """Which dataset versions and models used this speaker's recordings."""
    ledger_recordings = 0
    if ledger_path(data_dir).is_file():
        with Ledger(ledger_path(data_dir)) as ledger:
            ledger_recordings = len(_speaker_ledger_rows(ledger, speaker_id))
    datasets = _speaker_datasets(data_dir, speaker_id)
    versions = {d["dataset_version"] for d in datasets}
    return {
        "speaker_id": speaker_id,
        "ledger_recordings": ledger_recordings,
        "datasets": datasets,
        "models_dir_found": models_dir is not None and Path(models_dir).is_dir(),
        "models": [m for m in model_references(models_dir) if m["dataset_version"] in versions],
    }


def purge_speaker(data_dir, speaker_id, models_dir=None, *, apply=False):
    """Erase a speaker from the ledger, the audio cache and every dataset version; report affected models.

    Steps run audio first, ledger second, datasets third, each driven by its
    own state, so an interrupted purge is finished by running it again. An
    audio file is deleted only when no other active recording shares its
    hash. A dataset version containing the speaker keeps only SUPERSEDED.json.
    Interrupted dataset builds are removed too: they may hold copies of the
    audio and cannot be attributed to a speaker. Without `apply` nothing changes.
    """
    ledger_recordings, audio_files, audio_shared = 0, 0, 0
    if ledger_path(data_dir).is_file():
        with Ledger(ledger_path(data_dir)) as ledger:
            rows = _speaker_ledger_rows(ledger, speaker_id)
            ids = [row["recording_id"] for row in rows]
            hashes = {row["audio_sha256"] for row in rows} - {None}
            erasable = [sha for sha in sorted(hashes) if not ledger.audio_hash_in_use(sha, exclude_ids=ids)]
            audio_shared = len(hashes) - len(erasable)
            audio_files = sum(1 for sha in erasable if audio_path(data_dir, sha).exists())
            ledger_recordings = len(ids)
            if apply:
                for sha in erasable:
                    audio_path(data_dir, sha).unlink(missing_ok=True)
                ledger.mark_removed(ids)

    datasets = _speaker_datasets(data_dir, speaker_id)
    partial = [path for path, state in _dataset_dirs(data_dir) if state == "partial"]
    if apply:
        now = utc_iso()
        for entry in datasets:
            _supersede(Path(entry["path"]), entry["dataset_version"], entry["recordings"], now)
        for path in partial:
            shutil.rmtree(path)

    review = {d["dataset_version"] for d in datasets} | _superseded_versions(data_dir)
    return {
        "applied": apply,
        "ledger_recordings": ledger_recordings,
        "audio_files": audio_files,
        "audio_files_shared": audio_shared,
        "datasets": [{"dataset_version": d["dataset_version"], "recordings": d["recordings"]} for d in datasets],
        "interrupted_builds": len(partial),
        "models_dir_found": models_dir is not None and Path(models_dir).is_dir(),
        "models": [m for m in model_references(models_dir) if m["dataset_version"] in review],
    }


def _supersede(version_dir, version, purged_recordings, now):
    """Replace a dataset version's files with the marker. The marker goes first so a crash keeps the record."""
    marker = {
        "dataset_version": version,
        "superseded_at": now,
        "reason": "speaker purge",
        "purged_recordings": purged_recordings,
    }
    tmp = version_dir / (SUPERSEDED_FILENAME + ".tmp")
    tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    os.replace(tmp, version_dir / SUPERSEDED_FILENAME)
    for child in version_dir.iterdir():
        if child.name == SUPERSEDED_FILENAME:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def prune(data_dir, *, keep_datasets=None, models_dir=None, include_referenced=False, runs_dir=None, apply=False):
    """Retention. Keeps the newest `keep_datasets` dataset versions and deletes the rest, sparing versions
    that a model under `models_dir` still names unless `include_referenced` (that erases the record of
    which speakers such a model was trained on). With `runs_dir`, deletes the intermediate `checkpoint-N`
    directories of finished training runs (they can memorise training utterances), keeping the final model.
    Without `apply` nothing changes."""
    report = {
        "applied": apply,
        "keep_datasets": keep_datasets,
        "runs_dir": None if runs_dir is None else str(runs_dir),
        "datasets_total": 0,
        "datasets": [],
        "datasets_protected": [],
        "checkpoints": [],
        "checkpoints_in_unfinished_runs": 0,
        "models_dir_found": models_dir is not None and Path(models_dir).is_dir(),
    }

    if keep_datasets is not None:
        built = []
        for path, state in _dataset_dirs(data_dir):
            if state == "built":
                metadata = _load_json(path / BUILD_METADATA_FILENAME)
                created_at = metadata.get("created_at") if isinstance(metadata, dict) else None
                built.append((created_at or "", path.name, path))
        report["datasets_total"] = len(built)
        built.sort(reverse=True)
        referenced = {}
        for ref in model_references(models_dir):
            referenced.setdefault(ref["dataset_version"], []).append(ref["path"])
        for created_at, version, path in built[keep_datasets:]:
            if version in referenced and not include_referenced:
                report["datasets_protected"].append({"dataset_version": version, "models": referenced[version]})
                continue
            report["datasets"].append({"dataset_version": version, "created_at": created_at, "bytes": _tree_size(path)})
            if apply:
                shutil.rmtree(path)

    if runs_dir is not None and Path(runs_dir).is_dir():
        for path in sorted(Path(runs_dir).rglob("checkpoint-*")):
            if not path.is_dir() or not _CHECKPOINT.fullmatch(path.name):
                continue
            if not (path.parent / RUN_DONE_FILENAME).is_file():
                report["checkpoints_in_unfinished_runs"] += 1
                continue
            report["checkpoints"].append({"path": str(path), "bytes": _tree_size(path)})
            if apply:
                shutil.rmtree(path)
    return report


def _mb(size):
    return f"{size / 1e6:.1f} MB"


def _model_lines(models, models_dir_found):
    if not models_dir_found:
        return ["Models: --models-dir not found, so no model or training run was checked"]
    lines = [f"Models and training runs trained on these dataset versions: {len(models)}"]
    lines += [f"  {m['path']}  (dataset_version {m['dataset_version']}, {m['metadata_file']})" for m in models]
    return lines


def render_lineage(report):
    lines = [
        f"Speaker {report['speaker_id']}",
        f"Active ledger recordings: {report['ledger_recordings']}",
        f"Dataset versions containing this speaker: {len(report['datasets'])}",
    ]
    lines += [
        f"  {d['dataset_version']}  {d['recordings']} recordings  splits: {', '.join(d['splits'])}"
        for d in report["datasets"]
    ]
    return lines + _model_lines(report["models"], report["models_dir_found"])


def render_purge(report):
    applied = report["applied"]
    lines = [] if applied else ["Dry run: nothing was changed. Re-run with --yes to apply."]
    lines.append(
        f"Ledger: {report['ledger_recordings']} active recordings {'tombstoned' if applied else 'would be tombstoned'}"
    )
    lines.append(
        f"Audio: {report['audio_files']} files {'deleted' if applied else 'would be deleted'}"
        f" ({report['audio_files_shared']} kept: still used by other recordings)"
    )
    lines.append(
        f"Dataset versions {'superseded' if applied else 'that would be superseded'} "
        f"(data files deleted, marker kept): {len(report['datasets'])}"
    )
    lines += [f"  {d['dataset_version']}  {d['recordings']} recordings" for d in report["datasets"]]
    if report["interrupted_builds"]:
        lines.append(
            f"Interrupted dataset builds {'removed' if applied else 'that would be removed'}: {report['interrupted_builds']}"
        )
    if not report["models_dir_found"]:
        lines.append("Models: --models-dir not found, so no model or training run was checked")
    else:
        lines.append(f"Models and training runs to retrain or retire (never modified automatically): {len(report['models'])}")
        lines += [f"  {m['path']}  (dataset_version {m['dataset_version']}): retrain or retire required" for m in report["models"]]
    return lines


def render_prune(report):
    applied = report["applied"]
    lines = [] if applied else ["Dry run: nothing was changed. Re-run with --yes to apply."]
    if report["keep_datasets"] is not None:
        size = sum(d["bytes"] for d in report["datasets"])
        lines.append(
            f"Dataset versions {'deleted' if applied else 'that would be deleted'}: {len(report['datasets'])} "
            f"of {report['datasets_total']}, newest {report['keep_datasets']} kept ({_mb(size)})"
        )
        lines += [f"  {d['dataset_version']}  created {d['created_at'] or 'unknown'}  {_mb(d['bytes'])}" for d in report["datasets"]]
        if report["datasets_protected"]:
            lines.append(
                f"Kept because a model names them (--include-referenced deletes them): {len(report['datasets_protected'])}"
            )
            lines += [f"  {d['dataset_version']}  used by {len(d['models'])} model(s)" for d in report["datasets_protected"]]
        if not report["models_dir_found"]:
            lines.append("Note: --models-dir not found, so no version was protected as referenced")
    if report["runs_dir"] is not None:
        size = sum(c["bytes"] for c in report["checkpoints"])
        lines.append(
            f"Checkpoint directories {'deleted' if applied else 'that would be deleted'}: {len(report['checkpoints'])} ({_mb(size)})"
        )
        lines += [f"  {c['path']}  {_mb(c['bytes'])}" for c in report["checkpoints"]]
        if report["checkpoints_in_unfinished_runs"]:
            lines.append(
                f"Left alone: {report['checkpoints_in_unfinished_runs']} checkpoint directories in runs without a "
                f"{RUN_DONE_FILENAME} (unfinished)"
            )
    return lines
