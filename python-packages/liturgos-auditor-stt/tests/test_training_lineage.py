import hashlib
import json
import logging
import subprocess
from types import SimpleNamespace

import pytest

from auditor_stt.training import lineage

CONFIG = {
    "language": "fi", "task": "transcribe", "learning_rate": 1e-5, "epochs": 3.0, "batch_size": 4,
    "gradient_accumulation_steps": 1, "eval_batch_size": 4, "warmup_ratio": 0.05, "seed": 42,
    "fp16": True, "bf16": False, "gradient_checkpointing": True,
}
PEFT = {"method": "lora", "r": 32, "alpha": 64, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"],
        "trainable_parameters": 6_500_000, "total_parameters": 815_000_000}


def _dataset_dir(tmp_path, version="v-abc123", manifest=b'{"recordings": [1, 2, 3]}'):
    if version is not None:
        (tmp_path / "build_metadata.json").write_text(json.dumps({"dataset_version": version, "seed": 42}))
    if manifest is not None:
        (tmp_path / "manifest.json").write_bytes(manifest)
    return tmp_path


def _build(dataset_dir, **overrides):
    kwargs = dict(
        model_id="openai/whisper-small", preset="small", base_model_revision="rev123",
        dataset_dir=dataset_dir, config=CONFIG, peft=PEFT,
        sizes={"train": 90, "dev": 5, "test": 5}, train_metrics={"train_loss": 1.5},
    )
    kwargs.update(overrides)
    return lineage.build_metadata(**kwargs)


# --- dataset_lineage ------------------------------------------------------------

def test_dataset_lineage_reads_version_and_hashes_the_manifest(tmp_path):
    manifest = b'{"recordings": [1, 2, 3]}'
    result = lineage.dataset_lineage(_dataset_dir(tmp_path, manifest=manifest))
    assert result == {
        "dataset_version": "v-abc123",
        "dataset_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def test_dataset_lineage_is_null_when_the_files_are_absent(tmp_path):
    assert lineage.dataset_lineage(tmp_path) == {"dataset_version": None, "dataset_manifest_sha256": None}


def test_dataset_lineage_version_and_manifest_are_independent(tmp_path):
    only_manifest = lineage.dataset_lineage(_dataset_dir(tmp_path, version=None))
    assert only_manifest["dataset_version"] is None and only_manifest["dataset_manifest_sha256"]


def test_dataset_lineage_older_build_metadata_without_a_version_gives_null(tmp_path):
    (tmp_path / "build_metadata.json").write_text(json.dumps({"seed": 42, "speaker_disjoint": True}))
    assert lineage.dataset_lineage(tmp_path)["dataset_version"] is None


def test_dataset_lineage_survives_corrupt_build_metadata(tmp_path, caplog):
    (tmp_path / "build_metadata.json").write_text("{not json")
    with caplog.at_level(logging.WARNING, logger=lineage.logger.name):
        assert lineage.dataset_lineage(tmp_path)["dataset_version"] is None
    assert "build_metadata.json" in caplog.text


# --- git_commit -------------------------------------------------------------------

def test_git_commit_returns_the_trimmed_head(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="0123abcd\n")

    monkeypatch.setattr(lineage.subprocess, "run", fake_run)
    assert lineage.git_commit() == "0123abcd"
    assert calls[0][0] == ["git", "rev-parse", "HEAD"]
    assert calls[0][1]["check"] is True


@pytest.mark.parametrize("failure", [
    FileNotFoundError("git"),
    subprocess.CalledProcessError(128, "git"),
    subprocess.TimeoutExpired("git", 10),
])
def test_git_commit_is_none_when_git_is_unavailable_or_fails(monkeypatch, failure):
    def fake_run(cmd, **kwargs):
        raise failure

    monkeypatch.setattr(lineage.subprocess, "run", fake_run)
    assert lineage.git_commit() is None


def test_git_commit_uses_env_variable_when_set(monkeypatch):
    """AUDITOR_STT_GIT_COMMIT environment variable takes precedence over git command."""
    monkeypatch.setenv("AUDITOR_STT_GIT_COMMIT", "envcommit123")

    # Even if git would work, env var is used
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(stdout="gitcommit456\n")

    monkeypatch.setattr(lineage.subprocess, "run", fake_run)
    assert lineage.git_commit() == "envcommit123"


def test_git_commit_strips_env_variable(monkeypatch):
    """AUDITOR_STT_GIT_COMMIT is stripped of whitespace."""
    monkeypatch.setenv("AUDITOR_STT_GIT_COMMIT", "  envcommit123  ")
    assert lineage.git_commit() == "envcommit123"


def test_git_commit_ignores_empty_env_variable(monkeypatch):
    """Empty AUDITOR_STT_GIT_COMMIT falls back to git command."""
    monkeypatch.setenv("AUDITOR_STT_GIT_COMMIT", "")

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(stdout="gitcommit456\n")

    monkeypatch.setattr(lineage.subprocess, "run", fake_run)
    assert lineage.git_commit() == "gitcommit456"


# --- package_versions ---------------------------------------------------------------

def test_package_versions_maps_missing_packages_to_none():
    versions = lineage.package_versions(("pytest", "auditor-stt-no-such-package"))
    assert versions["pytest"]
    assert versions["auditor-stt-no-such-package"] is None


def test_tracked_packages_cover_the_training_stack():
    assert set(lineage.TRACKED_PACKAGES) == {"torch", "transformers", "peft", "datasets", "accelerate"}


# --- build_metadata -------------------------------------------------------------------

def test_metadata_keeps_the_existing_keys_and_adds_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage, "git_commit", lambda: "deadbeef")
    dataset_dir = _dataset_dir(tmp_path)
    metadata = _build(dataset_dir)

    assert metadata["base_model"] == "openai/whisper-small"
    assert metadata["language"] == "fi" and metadata["task"] == "transcribe"
    assert metadata["dataset"] == str(dataset_dir.resolve())
    assert metadata["epochs"] == 3.0 and metadata["learning_rate"] == 1e-5 and metadata["seed"] == 42
    assert (metadata["train_examples"], metadata["dev_examples"], metadata["test_examples"]) == (90, 5, 5)
    assert metadata["train_metrics"] == {"train_loss": 1.5}

    assert metadata["dataset_version"] == "v-abc123"
    assert metadata["dataset_manifest_sha256"] == lineage.sha256_file(dataset_dir / "manifest.json")
    assert metadata["base_model_revision"] == "rev123"
    assert metadata["git_commit"] == "deadbeef"
    assert set(metadata["package_versions"]) == set(lineage.TRACKED_PACKAGES)
    assert metadata["preset"] == "small"
    assert metadata["config"] == CONFIG
    assert metadata["peft"] == PEFT


def test_metadata_lineage_is_null_without_manifest_git_or_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage, "git_commit", lambda: None)
    metadata = _build(_dataset_dir(tmp_path, version=None, manifest=None), base_model_revision=None,
                      preset=None, peft={"method": "none"})
    assert metadata["dataset_version"] is None
    assert metadata["dataset_manifest_sha256"] is None
    assert metadata["base_model_revision"] is None
    assert metadata["git_commit"] is None
    assert metadata["preset"] is None
    assert metadata["peft"] == {"method": "none"}


def test_metadata_is_json_serialisable(tmp_path):
    json.dumps(_build(_dataset_dir(tmp_path)), ensure_ascii=False, default=str)
