import json
import os
from pathlib import Path

import pytest

from auditor_stt.cli import main
from auditor_stt.dataset.lineage import model_references
from auditor_stt.serve import registry
from auditor_stt.serve.registry import (
    GateRefusedError,
    RegistryError,
    current_version,
    gate_summary,
    list_models,
    promote,
    read_gate,
    register,
    resolve,
    show,
)

PASSED_GATE = {
    "schema": 1,
    "passed": True,
    "dataset_version": "ds-gate",
    "checks": [
        {"name": "beats_baseline", "passed": True, "detail": "candidate 0.10 vs baseline 0.20"},
        {"name": "longform_no_regression", "passed": None, "detail": "skipped"},
    ],
    "metrics": {},
}
FAILED_GATE = {**PASSED_GATE, "passed": False}
TRAINING = {"base_model": "openai/whisper-small", "dataset_version": "ds-train", "train_examples": 1234, "seed": 42}


def _write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def _export(tmp_path, name="export", gate=None, merged=True):
    """A stand-in CTranslate2 export directory, with the huge `_merged_hf` that `--merge-lora` leaves inside."""
    export = tmp_path / name
    (export / "vocabulary").mkdir(parents=True)
    (export / "model.bin").write_bytes(b"weights")
    (export / "config.json").write_text("{}", encoding="utf-8")
    (export / "vocabulary" / "vocab.json").write_text("{}", encoding="utf-8")
    if merged:
        (export / "_merged_hf").mkdir()
        (export / "_merged_hf" / "model.safetensors").write_bytes(b"x" * 100)
    if gate is not None:
        _write_json(export / "gate.json", gate)
    return export


def _run(tmp_path, training=None):
    return _write_json(tmp_path / "run" / "training_metadata.json", training or TRAINING).parent


@pytest.fixture
def reg(tmp_path):
    return tmp_path / "registry"


def _registered(reg, tmp_path, version="v1", gate=PASSED_GATE, name="whisper-fi"):
    export = _export(tmp_path, name=f"export-{name}-{version}", gate=gate)
    register(reg, name, version, export)
    return reg / name / version


# --- register -----------------------------------------------------------------------


def test_register_copies_the_export_without_merged_hf_and_the_gate(reg, tmp_path):
    export = _export(tmp_path, gate=PASSED_GATE)

    register(reg, "whisper-fi", "v1", export)

    version_dir = reg / "whisper-fi" / "v1"
    assert (version_dir / "ct2" / "model.bin").read_bytes() == b"weights"
    assert (version_dir / "ct2" / "config.json").is_file()
    assert (version_dir / "ct2" / "vocabulary" / "vocab.json").is_file()
    assert not (version_dir / "ct2" / "_merged_hf").exists()
    assert not (version_dir / "ct2" / "gate.json").exists()
    assert json.loads((version_dir / "gate.json").read_text(encoding="utf-8")) == PASSED_GATE
    assert sorted(p.name for p in version_dir.iterdir()) == ["ct2", "gate.json", "metadata.json"]
    assert (export / "_merged_hf" / "model.safetensors").is_file()  # the source is left alone
    assert (export / "model.bin").is_file()


def test_only_a_top_level_merged_hf_and_gate_are_excluded(reg, tmp_path):
    export = _export(tmp_path, merged=False)
    (export / "vocabulary" / "_merged_hf").mkdir()
    (export / "vocabulary" / "gate.json").write_text("{}", encoding="utf-8")

    register(reg, "whisper-fi", "v1", export)

    assert (reg / "whisper-fi" / "v1" / "ct2" / "vocabulary" / "_merged_hf").is_dir()
    assert (reg / "whisper-fi" / "v1" / "ct2" / "vocabulary" / "gate.json").is_file()


def test_metadata_has_the_documented_top_level_keys(reg, tmp_path):
    export = _export(tmp_path, gate=PASSED_GATE)
    run = _run(tmp_path)

    returned = register(reg, "whisper-fi", "v1", export, training_run_dir=run, quantization="int8")

    stored = json.loads((reg / "whisper-fi" / "v1" / "metadata.json").read_text(encoding="utf-8"))
    assert stored == returned
    assert stored["name"] == "whisper-fi" and stored["version"] == "v1"
    assert stored["created_at"].endswith("+00:00")
    assert stored["dataset_version"] == "ds-gate"
    assert stored["base_model"] == "openai/whisper-small"
    assert stored["quantization"] == "int8"
    assert stored["source"] == str(export)
    assert stored["gate"] == {"passed": True, "path": "gate.json"}
    assert stored["training_run"]["path"] == str(run)
    assert stored["training_run"]["train_examples"] == 1234


def test_metadata_records_absolute_source_and_run_paths_even_when_given_relative(reg, tmp_path, monkeypatch):
    export = _export(tmp_path, gate=PASSED_GATE)
    run = _run(tmp_path)
    monkeypatch.chdir(tmp_path)

    metadata = register(
        reg, "whisper-fi", "v1", export.relative_to(tmp_path), training_run_dir=run.relative_to(tmp_path)
    )

    assert Path(metadata["source"]).is_absolute() and Path(metadata["source"]) == export
    assert Path(metadata["training_run"]["path"]).is_absolute() and Path(metadata["training_run"]["path"]) == run


def test_metadata_without_gate_or_training_run(reg, tmp_path):
    metadata = register(reg, "whisper-fi", "v1", _export(tmp_path))

    assert metadata["dataset_version"] is None
    assert metadata["base_model"] is None
    assert metadata["gate"] is None
    assert metadata["training_run"] is None
    assert "quantization" not in metadata  # only when known
    assert not (reg / "whisper-fi" / "v1" / "gate.json").exists()


def test_a_failed_gate_is_recorded_as_such(reg, tmp_path):
    metadata = register(reg, "whisper-fi", "v1", _export(tmp_path, gate=FAILED_GATE))
    assert metadata["gate"] == {"passed": False, "path": "gate.json"}


@pytest.mark.parametrize(
    ("gate", "training", "expected"),
    [
        ({**PASSED_GATE, "dataset_version": "from-gate"}, {**TRAINING, "dataset_version": "from-run"}, "from-gate"),
        ({**PASSED_GATE, "dataset_version": None}, {**TRAINING, "dataset_version": "from-run"}, "from-run"),
        (None, {**TRAINING, "dataset_version": "from-run"}, "from-run"),
        ({**PASSED_GATE, "dataset_version": "from-gate"}, None, "from-gate"),
        ({**PASSED_GATE, "dataset_version": None}, {"base_model": "x"}, None),
        (None, None, None),
    ],
)
def test_dataset_version_prefers_the_gate_then_the_training_run(reg, tmp_path, gate, training, expected):
    export = _export(tmp_path, gate=gate)
    run = _run(tmp_path, training) if training is not None else None

    assert register(reg, "whisper-fi", "v1", export, training_run_dir=run)["dataset_version"] == expected


def test_quantization_comes_from_the_argument_then_the_gate_then_the_training_run(reg, tmp_path):
    with_gate = _export(tmp_path, name="a", gate={**PASSED_GATE, "quantization": "float16"})
    assert register(reg, "m", "v1", with_gate)["quantization"] == "float16"
    assert register(reg, "m", "v2", with_gate, quantization="int8")["quantization"] == "int8"
    run = _run(tmp_path, {**TRAINING, "quantization": "int8_float16"})
    assert register(reg, "m", "v3", _export(tmp_path, name="b"), training_run_dir=run)["quantization"] == "int8_float16"


def test_training_run_leaves_out_large_values(reg, tmp_path):
    run = _run(tmp_path, {**TRAINING, "train_metrics": {"log": ["x" * 100] * 200}, "peft": {"r": 8}})

    summary = register(reg, "whisper-fi", "v1", _export(tmp_path), training_run_dir=run)["training_run"]

    assert "train_metrics" not in summary
    assert summary["peft"] == {"r": 8}
    assert summary["train_examples"] == 1234


def test_a_training_run_without_metadata_is_an_error_and_registers_nothing(reg, tmp_path):
    empty_run = tmp_path / "empty-run"
    empty_run.mkdir()
    with pytest.raises(RegistryError, match="training_metadata.json"):
        register(reg, "whisper-fi", "v1", _export(tmp_path), training_run_dir=empty_run)
    assert not (reg / "whisper-fi").exists()


def test_an_explicit_gate_path_wins_over_the_gate_in_the_export(reg, tmp_path):
    export = _export(tmp_path, gate=FAILED_GATE)
    outside = _write_json(tmp_path / "elsewhere" / "my-gate.json", PASSED_GATE)

    metadata = register(reg, "whisper-fi", "v1", export, gate_path=outside)

    assert metadata["gate"]["passed"] is True
    assert read_gate(reg, "whisper-fi", "v1") == PASSED_GATE


def test_a_missing_or_unreadable_explicit_gate_is_an_error(reg, tmp_path):
    export = _export(tmp_path)
    with pytest.raises(RegistryError, match="does not exist"):
        register(reg, "whisper-fi", "v1", export, gate_path=tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(RegistryError, match="not a readable JSON object"):
        register(reg, "whisper-fi", "v1", export, gate_path=broken)
    assert list_models(reg) == []


def test_register_refuses_to_overwrite_a_version(reg, tmp_path):
    first = _export(tmp_path, name="first")
    register(reg, "whisper-fi", "v1", first)

    other = _export(tmp_path, name="second")
    (other / "model.bin").write_bytes(b"different")
    with pytest.raises(RegistryError, match="already registered"):
        register(reg, "whisper-fi", "v1", other)

    assert (reg / "whisper-fi" / "v1" / "ct2" / "model.bin").read_bytes() == b"weights"


def test_register_rejects_a_directory_that_is_not_a_ct2_export(reg, tmp_path):
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(RegistryError, match="no model.bin"):
        register(reg, "whisper-fi", "v1", tmp_path / "hf")
    with pytest.raises(RegistryError, match="does not exist"):
        register(reg, "whisper-fi", "v1", tmp_path / "missing")
    assert not reg.exists()


def test_a_copy_that_fails_leaves_nothing_behind(reg, tmp_path, monkeypatch):
    def broken_copytree(src, dst, **kwargs):
        Path(dst).mkdir()
        (Path(dst) / "half.bin").write_bytes(b"x")
        raise OSError("disk full")

    monkeypatch.setattr(registry.shutil, "copytree", broken_copytree)
    export = _export(tmp_path, gate=PASSED_GATE)

    with pytest.raises(OSError, match="disk full"):
        register(reg, "whisper-fi", "v1", export, move=True)

    assert list((reg / "whisper-fi").iterdir()) == []  # neither a version nor a staging directory
    assert (export / "model.bin").is_file()  # a failed move never loses the export


@pytest.mark.parametrize(
    "bad",
    ["../x", "..", ".", "", "a/b", "a\\b", ".hidden", "-lead", "with space", "x" * 65, "a:b", "a@b", "ä"],
)
def test_names_and_versions_are_validated_before_any_path_is_built(reg, tmp_path, bad):
    export = _export(tmp_path)
    with pytest.raises(RegistryError, match="Invalid model name"):
        register(reg, bad, "v1", export)
    with pytest.raises(RegistryError, match="Invalid version"):
        register(reg, "whisper-fi", bad, export)
    with pytest.raises(RegistryError):
        promote(reg, bad, "v1")
    with pytest.raises(RegistryError):
        promote(reg, "whisper-fi", bad)
    assert not reg.exists()
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize("good", ["v1", "2026-05-01", "whisper_fi.v2", "A", "0"])
def test_reasonable_names_are_accepted(reg, tmp_path, good):
    register(reg, good, good, _export(tmp_path))
    assert (reg / good / good / "ct2" / "model.bin").is_file()


def test_move_removes_what_was_copied_but_keeps_merged_hf(reg, tmp_path):
    export = _export(tmp_path, gate=PASSED_GATE)

    register(reg, "whisper-fi", "v1", export, move=True)

    assert sorted(p.name for p in export.iterdir()) == ["_merged_hf"]  # the checkpoint was never the registry's
    assert (reg / "whisper-fi" / "v1" / "ct2" / "model.bin").is_file()
    assert read_gate(reg, "whisper-fi", "v1") == PASSED_GATE


def test_move_removes_the_export_directory_when_nothing_is_left(reg, tmp_path):
    export = _export(tmp_path, merged=False)
    register(reg, "whisper-fi", "v1", export, move=True)
    assert not export.exists()


def test_move_leaves_a_gate_it_did_not_register(reg, tmp_path):
    export = _export(tmp_path, gate=FAILED_GATE, merged=False)
    outside = _write_json(tmp_path / "other-gate.json", PASSED_GATE)

    register(reg, "whisper-fi", "v1", export, gate_path=outside, move=True)

    assert [p.name for p in export.iterdir()] == ["gate.json"]
    assert outside.is_file()


def test_lineage_tool_finds_the_dataset_version_in_registry_metadata(reg, tmp_path):
    version_dir = _registered(reg, tmp_path)

    found = model_references(reg)

    assert [(Path(f["path"]), f["dataset_version"], f["metadata_file"]) for f in found] == [
        (version_dir, "ds-gate", "metadata.json")
    ]


# --- promote --------------------------------------------------------------------------


def _current(reg, name="whisper-fi"):
    return json.loads((reg / name / "current.json").read_text(encoding="utf-8"))


def test_promote_a_version_that_passed(reg, tmp_path):
    _registered(reg, tmp_path)

    result = promote(reg, "whisper-fi", "v1")

    assert _current(reg) == result
    assert result["version"] == "v1"
    assert result["forced"] is False
    assert result["promoted_by"] == "cli"
    assert result["promoted_at"].endswith("+00:00")
    assert current_version(reg, "whisper-fi") == "v1"


def test_promote_records_who_promoted(reg, tmp_path):
    _registered(reg, tmp_path)
    assert promote(reg, "whisper-fi", "v1", promoted_by="ops")["promoted_by"] == "ops"


def test_promote_refuses_a_failed_gate_and_changes_nothing(reg, tmp_path):
    _registered(reg, tmp_path, "v1", gate=PASSED_GATE)
    promote(reg, "whisper-fi", "v1")
    _registered(reg, tmp_path, "v2", gate=FAILED_GATE)

    with pytest.raises(GateRefusedError, match="did not pass"):
        promote(reg, "whisper-fi", "v2")

    assert current_version(reg, "whisper-fi") == "v1"


def test_promote_refuses_a_version_without_a_gate(reg, tmp_path):
    _registered(reg, tmp_path, gate=None)
    with pytest.raises(GateRefusedError, match="no readable gate.json"):
        promote(reg, "whisper-fi", "v1")
    assert not (reg / "whisper-fi" / "current.json").exists()


@pytest.mark.parametrize("passed", ["true", 1, "yes", None, [True]])
def test_only_a_boolean_true_counts_as_passed(reg, tmp_path, passed):
    _registered(reg, tmp_path, gate={**PASSED_GATE, "passed": passed})
    with pytest.raises(GateRefusedError):
        promote(reg, "whisper-fi", "v1")


def test_promote_refuses_an_unreadable_gate(reg, tmp_path):
    version_dir = _registered(reg, tmp_path)
    (version_dir / "gate.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(GateRefusedError):
        promote(reg, "whisper-fi", "v1")
    assert read_gate(reg, "whisper-fi", "v1") is None


def test_force_promotes_past_a_failed_gate_and_records_it(reg, tmp_path):
    _registered(reg, tmp_path, gate=FAILED_GATE)
    result = promote(reg, "whisper-fi", "v1", force=True)
    assert result["forced"] is True
    assert _current(reg)["forced"] is True


def test_force_promotes_a_version_without_a_gate_and_records_it(reg, tmp_path):
    _registered(reg, tmp_path, gate=None)
    assert promote(reg, "whisper-fi", "v1", force=True)["forced"] is True


def test_force_on_a_passed_gate_is_not_recorded_as_forced(reg, tmp_path):
    _registered(reg, tmp_path)
    assert promote(reg, "whisper-fi", "v1", force=True)["forced"] is False


def test_promote_an_unregistered_version_is_an_error(reg, tmp_path):
    _registered(reg, tmp_path)
    with pytest.raises(RegistryError, match="not registered"):
        promote(reg, "whisper-fi", "v9")
    with pytest.raises(RegistryError, match="not registered"):
        promote(reg, "nobody", "v1", force=True)


def test_promoting_another_version_moves_the_pointer_back_and_forth(reg, tmp_path):
    _registered(reg, tmp_path, "v1")
    _registered(reg, tmp_path, "v2")
    promote(reg, "whisper-fi", "v2")
    promote(reg, "whisper-fi", "v1")  # a rollback is just a promotion
    assert current_version(reg, "whisper-fi") == "v1"


def test_current_json_is_replaced_atomically(reg, tmp_path, monkeypatch):
    _registered(reg, tmp_path, "v1")
    _registered(reg, tmp_path, "v2")
    promote(reg, "whisper-fi", "v1")
    before = (reg / "whisper-fi" / "current.json").read_text(encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError("interrupted")

    monkeypatch.setattr(registry.os, "replace", failing_replace)
    with pytest.raises(OSError, match="interrupted"):
        promote(reg, "whisper-fi", "v2")
    monkeypatch.undo()

    # The pointer still has its old, complete content, and no temp file is left next to it.
    assert (reg / "whisper-fi" / "current.json").read_text(encoding="utf-8") == before
    assert sorted(p.name for p in (reg / "whisper-fi").iterdir()) == ["current.json", "v1", "v2"]


def test_a_successful_promotion_leaves_no_temp_files(reg, tmp_path):
    _registered(reg, tmp_path)
    promote(reg, "whisper-fi", "v1")
    assert sorted(p.name for p in (reg / "whisper-fi").iterdir()) == ["current.json", "v1"]


def test_gate_summary_lists_the_verdict_and_every_check():
    assert gate_summary(PASSED_GATE) == [
        "Gate: PASSED",
        "  dataset_version: ds-gate",
        "  [PASS] beats_baseline: candidate 0.10 vs baseline 0.20",
        "  [SKIP] longform_no_regression: skipped",
    ]
    assert gate_summary(FAILED_GATE)[0] == "Gate: NOT PASSED"
    assert gate_summary(None) == ["Gate: none (no readable gate.json)"]
    assert gate_summary({"passed": True, "checks": ["junk", {"no": "name"}]}) == ["Gate: PASSED"]


# --- resolve --------------------------------------------------------------------------


def test_anything_that_is_not_a_registry_spec_is_returned_unchanged(reg):
    assert resolve("large-v3-turbo", reg) == ("large-v3-turbo", "large-v3-turbo")
    assert resolve("/models/my-ct2", reg) == ("/models/my-ct2", "/models/my-ct2")
    assert resolve("Systran/faster-whisper-small", reg) == ("Systran/faster-whisper-small", "Systran/faster-whisper-small")
    assert not reg.exists()


def test_registry_name_resolves_to_the_current_version(reg, tmp_path):
    _registered(reg, tmp_path, "v1")
    _registered(reg, tmp_path, "v2")
    promote(reg, "whisper-fi", "v2")

    path, label = resolve("registry:whisper-fi", reg)

    assert Path(path) == reg / "whisper-fi" / "v2" / "ct2"
    assert label == "registry:whisper-fi@v2"


def test_registry_name_at_version_resolves_to_that_version_without_a_promotion(reg, tmp_path):
    _registered(reg, tmp_path, "v1")
    _registered(reg, tmp_path, "v2")
    promote(reg, "whisper-fi", "v2")

    path, label = resolve("registry:whisper-fi@v1", reg)

    assert Path(path) == reg / "whisper-fi" / "v1" / "ct2"
    assert label == "registry:whisper-fi@v1"
    assert resolve("registry:whisper-fi@v1", reg / ".." / reg.name)[1] == label  # any spelling of the directory


def test_resolve_errors_are_clear(reg, tmp_path):
    _registered(reg, tmp_path, "v1")

    with pytest.raises(RegistryError, match="no promoted version"):
        resolve("registry:whisper-fi", reg)
    with pytest.raises(RegistryError, match="whisper-fi@v9 is not registered"):
        resolve("registry:whisper-fi@v9", reg)
    with pytest.raises(RegistryError, match="nobody@v1 is not registered"):
        resolve("registry:nobody@v1", reg)
    with pytest.raises(RegistryError, match="no promoted version"):
        resolve("registry:nobody", reg)
    with pytest.raises(RegistryError, match="empty version"):
        resolve("registry:whisper-fi@", reg)


@pytest.mark.parametrize(
    "spec",
    ["registry:", "registry:../x", "registry:..", "registry:a/b@v1", "registry:whisper-fi@../v1", "registry:x@a@b", "registry:@v1"],
)
def test_resolve_rejects_specs_that_could_leave_the_registry(reg, tmp_path, spec):
    _registered(reg, tmp_path)
    with pytest.raises(RegistryError, match="Invalid"):
        resolve(spec, reg)


def test_resolve_refuses_a_version_without_a_ct2_directory(reg, tmp_path):
    (reg / "whisper-fi" / "v1").mkdir(parents=True)
    with pytest.raises(RegistryError, match="not registered"):
        resolve("registry:whisper-fi@v1", reg)


def test_a_dangling_current_pointer_is_an_error_not_a_crash(reg, tmp_path):
    _registered(reg, tmp_path)
    _write_json(reg / "whisper-fi" / "current.json", {"version": "v7"})
    with pytest.raises(RegistryError, match="whisper-fi@v7 is not registered"):
        resolve("registry:whisper-fi", reg)
    (reg / "whisper-fi" / "current.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(RegistryError, match="no promoted version"):
        resolve("registry:whisper-fi", reg)


# --- list / show ----------------------------------------------------------------------


def test_list_of_a_missing_or_empty_registry_is_empty(reg):
    assert list_models(reg) == []
    reg.mkdir()
    assert list_models(reg) == []


def test_list_reports_versions_and_the_current_one(reg, tmp_path):
    _registered(reg, tmp_path, "v1", name="alpha")
    _registered(reg, tmp_path, "v2", name="alpha")
    _registered(reg, tmp_path, "v1", name="beta")
    promote(reg, "alpha", "v2")

    assert list_models(reg) == [
        {"name": "alpha", "current_version": "v2", "versions": ["v1", "v2"]},
        {"name": "beta", "current_version": None, "versions": ["v1"]},
    ]


def test_versions_are_listed_oldest_first_not_alphabetically(reg, tmp_path):
    for version in ("v10", "v2", "v1"):
        _registered(reg, tmp_path, version)
    for version, created_at in (("v1", "2026-01-01"), ("v2", "2026-02-01"), ("v10", "2026-03-01")):
        path = reg / "whisper-fi" / version / "metadata.json"
        _write_json(path, {**json.loads(path.read_text(encoding="utf-8")), "created_at": created_at})

    assert list_models(reg)[0]["versions"] == ["v1", "v2", "v10"]


def test_list_ignores_staging_leftovers_and_strays(reg, tmp_path):
    _registered(reg, tmp_path)
    (reg / "whisper-fi" / ".v2-abcd1234.tmp" / "ct2").mkdir(parents=True)
    (reg / "whisper-fi" / "notes").mkdir()  # not a version: no ct2
    (reg / "stray.txt").write_text("x", encoding="utf-8")
    (reg / "empty-model").mkdir()

    assert list_models(reg) == [{"name": "whisper-fi", "current_version": None, "versions": ["v1"]}]


def test_show_defaults_to_the_current_version(reg, tmp_path):
    _registered(reg, tmp_path, "v1")
    _registered(reg, tmp_path, "v2")
    promote(reg, "whisper-fi", "v1")

    details = show(reg, "whisper-fi")

    assert details["version"] == "v1"
    assert details["versions"] == ["v1", "v2"]
    assert details["current"]["version"] == "v1"
    assert details["metadata"]["dataset_version"] == "ds-gate"
    assert details["gate"] == PASSED_GATE


def test_show_a_specific_version(reg, tmp_path):
    _registered(reg, tmp_path, "v1", gate=None)
    _registered(reg, tmp_path, "v2")
    promote(reg, "whisper-fi", "v2")

    details = show(reg, "whisper-fi", "v1")

    assert details["version"] == "v1"
    assert details["gate"] is None
    assert details["current"]["version"] == "v2"


def test_show_without_a_promotion_has_no_version_details(reg, tmp_path):
    _registered(reg, tmp_path)
    details = show(reg, "whisper-fi")
    assert details["current"] is None and details["version"] is None
    assert details["metadata"] is None and details["gate"] is None
    assert details["versions"] == ["v1"]


def test_show_errors(reg, tmp_path):
    _registered(reg, tmp_path)
    with pytest.raises(RegistryError, match="No model named"):
        show(reg, "nobody")
    with pytest.raises(RegistryError, match="no version 'v9'"):
        show(reg, "whisper-fi", "v9")
    with pytest.raises(RegistryError, match="Invalid"):
        show(reg, "../x")


def test_the_default_registry_dir_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDITOR_STT_REGISTRY_DIR", str(tmp_path / "from-env"))
    assert registry.default_registry_dir() == tmp_path / "from-env"
    monkeypatch.delenv("AUDITOR_STT_REGISTRY_DIR")
    assert registry.default_registry_dir() == Path("./models/registry")


# --- CLI ------------------------------------------------------------------------------


def _cli(reg, *argv):
    return main(["models", *argv, "--registry-dir", str(reg)])


def test_cli_register_promote_list_show(reg, tmp_path, capsys):
    export = _export(tmp_path, gate=PASSED_GATE)
    run = _run(tmp_path)

    assert _cli(reg, "register", "--name", "whisper-fi", "--version", "v1", "--export", str(export),
                "--training-run", str(run), "--quantization", "int8") == 0
    assert "Registered whisper-fi@v1 (dataset_version ds-gate, gate passed)" in capsys.readouterr().out

    assert _cli(reg, "promote", "--name", "whisper-fi", "--version", "v1") == 0
    out = capsys.readouterr().out
    assert "Gate: PASSED" in out and "[PASS] beats_baseline" in out
    assert "current version is now v1" in out and "forced" not in out

    assert _cli(reg, "list") == 0
    assert "whisper-fi  current: v1  versions: v1" in capsys.readouterr().out

    assert _cli(reg, "show", "--name", "whisper-fi") == 0
    out = capsys.readouterr().out
    assert "Current: v1, promoted" in out
    assert "dataset_version: ds-gate" in out and "base_model: openai/whisper-small" in out
    assert "quantization: int8" in out and "Gate: PASSED" in out


def test_cli_promote_exit_codes(reg, tmp_path, capsys):
    _registered(reg, tmp_path, "v1", gate=FAILED_GATE)
    _registered(reg, tmp_path, "v2", gate=None)

    assert _cli(reg, "promote", "--name", "whisper-fi", "--version", "v1") == 3
    captured = capsys.readouterr()
    assert "Gate: NOT PASSED" in captured.out and "did not pass" in captured.err
    assert _cli(reg, "promote", "--name", "whisper-fi", "--version", "v2") == 3
    assert "Gate: none" in capsys.readouterr().out
    assert not (reg / "whisper-fi" / "current.json").exists()

    assert _cli(reg, "promote", "--name", "whisper-fi", "--version", "v1", "--force") == 0
    assert "forced past the gate" in capsys.readouterr().out
    assert current_version(reg, "whisper-fi") == "v1"

    assert _cli(reg, "promote", "--name", "whisper-fi", "--version", "v9") == 1
    assert "not registered" in capsys.readouterr().err
    assert _cli(reg, "promote", "--name", "../x", "--version", "v1") == 1
    assert "Invalid model name" in capsys.readouterr().err


def test_cli_register_errors_and_move(reg, tmp_path, capsys):
    export = _export(tmp_path, gate=PASSED_GATE)
    args = ("register", "--name", "whisper-fi", "--version", "v1", "--export", str(export))

    assert _cli(reg, *args, "--move") == 0
    capsys.readouterr()
    assert [p.name for p in export.iterdir()] == ["_merged_hf"]

    assert _cli(reg, *args) == 1  # v1 exists (and the export is gone anyway)
    assert "already registered" in capsys.readouterr().err
    assert _cli(reg, "register", "--name", "x", "--version", "v1", "--export", str(tmp_path / "nowhere")) == 1
    assert "does not exist" in capsys.readouterr().err


def test_cli_list_and_show_on_an_empty_registry(reg, capsys):
    assert _cli(reg, "list") == 0
    assert "No models registered" in capsys.readouterr().out
    assert _cli(reg, "show", "--name", "whisper-fi") == 1
    assert "No model named" in capsys.readouterr().err


def test_cli_uses_the_registry_dir_from_the_environment(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AUDITOR_STT_REGISTRY_DIR", str(tmp_path / "env-registry"))
    export = _export(tmp_path, gate=PASSED_GATE)

    assert main(["models", "register", "--name", "m", "--version", "v1", "--export", str(export)]) == 0
    assert (tmp_path / "env-registry" / "m" / "v1" / "ct2" / "model.bin").is_file()
    assert main(["models", "list"]) == 0
    assert "m  current: none  versions: v1" in capsys.readouterr().out


def test_cli_output_carries_no_export_contents(reg, tmp_path, capsys):
    export = _export(tmp_path, gate=PASSED_GATE)
    (export / "notes.txt").write_text("secret words spoken by someone", encoding="utf-8")
    _cli(reg, "register", "--name", "whisper-fi", "--version", "v1", "--export", str(export))
    _cli(reg, "promote", "--name", "whisper-fi", "--version", "v1")
    _cli(reg, "list")
    _cli(reg, "show", "--name", "whisper-fi")
    assert "secret words" not in capsys.readouterr().out


def test_temp_names_are_hidden_from_the_name_pattern():
    # Staging directories start with a dot, which no registry name may.
    assert not registry._NAME.fullmatch(registry._temp_name(Path("v1")).name)
    assert os.path.basename(str(registry._temp_name(Path("dir") / "current.json"))).startswith(".current.json-")
