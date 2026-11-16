import json
import subprocess
from pathlib import Path

import pytest

from auditor_stt.training import export_onnx


def _source(tmp_path, name="model", adapter=False, mel=80):
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"max_target_positions": 448}))
    (path / "tokenizer.json").write_text("{}")
    (path / "preprocessor_config.json").write_text(json.dumps({"feature_size": mel}))
    (path / "generation_config.json").write_text(json.dumps({"suppress_tokens": [1]}))
    if adapter:
        (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/base"}))
    return path


class _Optimum:
    """Stands in for optimum-cli: records the command and writes what the real one writes."""

    def __init__(self, monkeypatch, files=("encoder_model.onnx", "decoder_model.onnx",
                                           "decoder_with_past_model.onnx", "decoder_model_merged.onnx")):
        self.calls = []
        self.files = files
        monkeypatch.setattr(export_onnx.shutil, "which", lambda name: "/bin/optimum-cli")
        monkeypatch.setattr(export_onnx.subprocess, "run", self._run)

    def _run(self, cmd, check):
        assert check is True
        output = Path(cmd[-1])
        assert not output.exists()
        self.calls.append(cmd)
        output.mkdir()
        for name in self.files:
            (output / name).write_bytes(b"onnx")
        # optimum writes the tokenizer and feature extractor next to the graphs
        (output / "tokenizer.json").write_text("{}")
        (output / "preprocessor_config.json").write_text(json.dumps({"feature_size": 80}))


def test_export_runs_the_with_past_task_and_leaves_an_engine_package(tmp_path, monkeypatch):
    optimum = _Optimum(monkeypatch)
    out = tmp_path / "out" / "pkg"

    assert export_onnx.export(_source(tmp_path), out) == out

    (cmd,) = optimum.calls
    assert cmd[1:3] == ["export", "onnx"]
    assert cmd[cmd.index("--task") + 1] == "automatic-speech-recognition-with-past"
    names = sorted(p.name for p in out.iterdir())
    # unmerged decoders are pruned: the engine only reads the merged graph
    assert "decoder_model.onnx" not in names and "decoder_with_past_model.onnx" not in names
    assert {"encoder_model.onnx", "decoder_model_merged.onnx", "tokenizer.json",
            "preprocessor_config.json", "generation_config.json", "config.json"} <= set(names)
    assert export_onnx.check_package(out) == 80


def test_adapter_without_merge_lora_is_refused_before_anything_runs(tmp_path, monkeypatch):
    optimum = _Optimum(monkeypatch)
    with pytest.raises(ValueError, match="--merge-lora"):
        export_onnx.export(_source(tmp_path, adapter=True), tmp_path / "pkg")
    assert optimum.calls == [] and not (tmp_path / "pkg").exists()


def test_adapter_is_merged_then_exported(tmp_path, monkeypatch):
    optimum = _Optimum(monkeypatch)
    merged = []

    def fake_merge(adapter_dir, merged_dir):
        merged.append(Path(merged_dir))
        for name in ("tokenizer.json", "preprocessor_config.json"):
            (Path(merged_dir) / name).write_text(json.dumps({"feature_size": 128}) if "prep" in name else "{}")
        (Path(merged_dir) / "generation_config.json").write_text("{}")

    monkeypatch.setattr(export_onnx, "_merge_adapter", fake_merge)
    out = tmp_path / "pkg"
    export_onnx.export(_source(tmp_path, adapter=True), out, merge_lora=True)

    assert Path(optimum.calls[0][optimum.calls[0].index("--model") + 1]) == merged[0]
    assert not merged[0].exists(), "the merged fp32 checkpoint is a temporary, not part of the package"
    assert (out / "generation_config.json").is_file()


def test_existing_output_and_missing_optimum_fail_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(export_onnx.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="optimum-cli"):
        export_onnx.export(_source(tmp_path), tmp_path / "pkg")
    _Optimum(monkeypatch)
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        export_onnx.export(_source(tmp_path, "m2"), tmp_path / "exists")


def test_export_without_a_merged_decoder_is_rejected(tmp_path, monkeypatch):
    _Optimum(monkeypatch, files=("encoder_model.onnx", "decoder_model.onnx"))
    with pytest.raises(export_onnx.PackageError, match="decoder_model_merged"):
        export_onnx.export(_source(tmp_path), tmp_path / "pkg")


def test_converter_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(export_onnx.shutil, "which", lambda name: "/bin/optimum-cli")

    def boom(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(export_onnx.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        export_onnx.export(_source(tmp_path), tmp_path / "pkg")


def test_missing_tokenizer_or_bad_mel_bins_fail_the_check(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "encoder_model_quantized.onnx").write_bytes(b"x")
    (pkg / "decoder_model_merged_quantized.onnx").write_bytes(b"x")
    with pytest.raises(export_onnx.PackageError, match="tokenizer.json"):
        export_onnx.check_package(pkg)
    (pkg / "tokenizer.json").write_text("{}")
    (pkg / "preprocessor_config.json").write_text(json.dumps({"feature_size": 64}))
    with pytest.raises(export_onnx.PackageError, match="80 or 128"):
        export_onnx.check_package(pkg)
    (pkg / "preprocessor_config.json").write_text(json.dumps({"feature_size": 128}))
    assert export_onnx.check_package(pkg, mel_bins=128) == 128
    with pytest.raises(export_onnx.PackageError, match="expected 80"):
        export_onnx.check_package(pkg, mel_bins=80)


def test_int8_replaces_fp32_graphs(tmp_path, monkeypatch):
    _Optimum(monkeypatch)
    calls = []

    def fake_quantize(directory):
        calls.append(directory)
        for stem in ("encoder_model", "decoder_model_merged"):
            (Path(directory) / f"{stem}_quantized.onnx").write_bytes(b"q")
            (Path(directory) / f"{stem}.onnx").unlink()

    monkeypatch.setattr(export_onnx, "_quantize_int8", fake_quantize)
    out = tmp_path / "pkg"
    export_onnx.export(_source(tmp_path), out, quantize="int8")
    assert calls == [out]
    assert (out / "encoder_model_quantized.onnx").is_file() and not (out / "encoder_model.onnx").exists()


def test_cli_export_onnx_reports_errors_without_a_traceback(tmp_path, monkeypatch, capsys):
    from auditor_stt import cli

    monkeypatch.setattr(export_onnx.shutil, "which", lambda name: None)
    rc = cli.main(["export-onnx", "--model", str(_source(tmp_path)), "--out", str(tmp_path / "pkg")])
    assert rc == 1
    assert "optimum-cli" in capsys.readouterr().err


def test_onnx_backend_is_selectable_in_eval():
    from auditor_stt.training import evaluate

    assert "onnx" in evaluate.BACKENDS
    with pytest.raises(export_onnx.PackageError):
        evaluate.onnx_backend("/nonexistent-dir")
