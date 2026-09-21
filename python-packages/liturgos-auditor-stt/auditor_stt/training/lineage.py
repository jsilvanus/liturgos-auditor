"""Lineage recorded in training_metadata.json: which data, code and libraries made a model.

Needed to answer "what was this model trained on?" later, e.g. when a
speaker's data is withdrawn. Everything here is best effort: a missing piece
is recorded as null and never stops a training run.
"""

import hashlib
import json
import logging
import subprocess
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = ("torch", "transformers", "peft", "datasets", "accelerate")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_lineage(dataset_dir):
    """dataset_version from build_metadata.json and the sha256 of manifest.json, null when absent.

    `dataset build` writes both; datasets built before that have neither.
    """
    dataset_dir = Path(dataset_dir)
    version = None
    build_metadata = dataset_dir / "build_metadata.json"
    if build_metadata.is_file():
        try:
            content = json.loads(build_metadata.read_text(encoding="utf-8"))
            version = content.get("dataset_version") if isinstance(content, dict) else None
        except ValueError:
            logger.warning("Ignoring unreadable %s", build_metadata)
    manifest = dataset_dir / "manifest.json"
    return {
        "dataset_version": version,
        "dataset_manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
    }


def git_commit(repo_dir=None):
    """HEAD of the checkout the training code runs from, or None (no git, or not a checkout)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir or Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def package_versions(names=TRACKED_PACKAGES):
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_metadata(*, model_id, preset, base_model_revision, dataset_dir, config, peft, sizes, train_metrics):
    """The training_metadata.json content. `config` holds every hyper-parameter, `peft` the adapter settings."""
    return {
        "base_model": model_id,
        "base_model_revision": base_model_revision,
        "preset": preset,
        "language": config["language"],
        "task": config["task"],
        "dataset": str(Path(dataset_dir).resolve()),
        **dataset_lineage(dataset_dir),
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "seed": config["seed"],
        "train_examples": sizes["train"],
        "dev_examples": sizes["dev"],
        "test_examples": sizes["test"],
        "train_metrics": train_metrics,
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "config": config,
        "peft": peft,
    }
