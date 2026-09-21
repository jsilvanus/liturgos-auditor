import json
import sys
import types
from pathlib import Path

import pytest

from auditor_stt.training import compat, export_ct2


def _model_dir(tmp_path, name="model", adapter=False, base="org/base-whisper"):
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text("{}")
    if adapter:
        (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    return path


class _Converter:
    """Stands in for ct2-transformers-converter: records the command, creates the output like the real one."""

    def __init__(self, monkeypatch, fail=False):
        self.calls = []
        self.fail = fail
        monkeypatch.setattr(export_ct2.shutil, "which", lambda name: "/bin/ct2-transformers-converter")
        monkeypatch.setattr(export_ct2.subprocess, "run", self._run)

    def _run(self, cmd, check):
        assert check is True
        model = Path(cmd[cmd.index("--model") + 1])
        output = Path(cmd[cmd.index("--output_dir") + 1])
        assert model.is_dir()
        assert not output.exists(), "ct2-transformers-converter refuses an existing output directory"
        self.calls.append({"model": model, "output": output, "cmd": cmd})
        if self.fail:
            raise export_ct2.subprocess.CalledProcessError(1, cmd)
        output.mkdir()
        (output / "model.bin").write_bytes(b"ct2")


@pytest.fixture
def merge(monkeypatch):
    """Replace the heavy merge with a fake that saves a marker file; returns its call log."""
    calls = []

    def fake_merge(adapter_dir, merged_dir):
        calls.append((Path(adapter_dir), Path(merged_dir)))
        (Path(merged_dir) / "merged.marker").write_text("merged")
        return Path(merged_dir)

    monkeypatch.setattr(export_ct2, "_merge_adapter", fake_merge)
    return calls


# --- merge decision ----------------------------------------------------------------

def test_adapter_dir_without_merge_lora_is_refused_before_anything_runs(tmp_path, monkeypatch, merge):
    converter = _Converter(monkeypatch)
    adapter = _model_dir(tmp_path, adapter=True)
    output = tmp_path / "ct2"

    with pytest.raises(ValueError, match="--merge-lora"):
        export_ct2.export(adapter, output)

    assert merge == [] and converter.calls == []
    assert not output.exists()


def test_adapter_dir_with_merge_lora_merges_then_converts_the_merged_checkpoint(tmp_path, monkeypatch, merge):
    converter = _Converter(monkeypatch)
    adapter = _model_dir(tmp_path, adapter=True)
    output = tmp_path / "out" / "ct2"

    assert export_ct2.export(adapter, output, quantization="int8", merge_lora=True) == output

    assert [call[0] for call in merge] == [adapter]
    (call,) = converter.calls
    assert call["model"] == merge[0][1] and call["model"] != adapter
    assert call["output"] == output
    assert call["cmd"][call["cmd"].index("--quantization") + 1] == "int8"
    # the merged checkpoint ends up inside the export, the scratch copy is gone
    assert (output / "model.bin").read_bytes() == b"ct2"
    assert (output / "_merged_hf" / "merged.marker").read_text() == "merged"
    assert not call["model"].exists()
    assert [p.name for p in output.parent.iterdir()] == ["ct2"]


def test_failed_conversion_cleans_up_the_scratch_merge_and_propagates(tmp_path, monkeypatch, merge):
    converter = _Converter(monkeypatch, fail=True)
    adapter = _model_dir(tmp_path, adapter=True)
    output = tmp_path / "ct2"

    with pytest.raises(export_ct2.subprocess.CalledProcessError):
        export_ct2.export(adapter, output, merge_lora=True)

    assert not converter.calls[0]["model"].exists()
    assert not output.exists()


def test_plain_model_dir_is_converted_directly_even_with_merge_lora(tmp_path, monkeypatch, merge):
    converter = _Converter(monkeypatch)
    model = _model_dir(tmp_path)

    for flag, name in ((False, "a"), (True, "b")):
        export_ct2.export(model, tmp_path / name, merge_lora=flag)

    assert merge == []
    assert [c["model"] for c in converter.calls] == [model, model]
    assert not (tmp_path / "a" / "_merged_hf").exists()


def test_plain_export_command_line_is_unchanged_for_a_complete_model_dir(tmp_path, monkeypatch):
    converter = _Converter(monkeypatch)
    model = _model_dir(tmp_path)
    for name in ("preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (model / name).write_text("{}")
    export_ct2.export(model, tmp_path / "ct2")

    assert converter.calls[0]["cmd"] == [
        "/bin/ct2-transformers-converter",
        "--model", str(model),
        "--output_dir", str(tmp_path / "ct2"),
        "--copy_files", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "--quantization", "float16",
    ]


def test_optional_files_that_a_model_dir_lacks_are_not_requested_but_required_ones_always_are(tmp_path, monkeypatch):
    # transformers 5 writes no special_tokens_map.json; a missing preprocessor_config.json
    # must still reach the converter so it fails loudly instead of shipping a wrong mel count.
    converter = _Converter(monkeypatch)
    model = _model_dir(tmp_path)
    (model / "tokenizer_config.json").write_text("{}")
    export_ct2.export(model, tmp_path / "ct2")

    cmd = converter.calls[0]["cmd"]
    copy_files = cmd[cmd.index("--copy_files") + 1:cmd.index("--quantization")]
    assert copy_files == ["preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"]


def test_existing_output_directory_is_refused(tmp_path, monkeypatch, merge):
    converter = _Converter(monkeypatch)
    model = _model_dir(tmp_path)
    (tmp_path / "ct2").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        export_ct2.export(model, tmp_path / "ct2")
    assert converter.calls == []


def test_missing_converter_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(export_ct2.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ct2-transformers-converter"):
        export_ct2.export(_model_dir(tmp_path), tmp_path / "ct2")


def test_main_passes_merge_lora_through(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(export_ct2, "export", lambda *args: seen.setdefault("args", args))
    export_ct2.main(["--model", "m", "--output", "o", "--merge-lora"])
    assert seen["args"] == ("m", "o", "float16", True)


# --- _merge_adapter with faked peft/transformers -------------------------------------

class _FakeMerged:
    generation_config = "base-generation-config"

    def __init__(self, log):
        self.log = log

    def save_pretrained(self, path):
        self.log.append(("save_model", path, self.generation_config))
        Path(path, "model.safetensors").write_bytes(b"w")


@pytest.fixture
def fake_libs(monkeypatch):
    log = []

    class PeftModel:
        @staticmethod
        def from_pretrained(base, adapter_path):
            log.append(("apply_adapter", base, adapter_path))
            return types.SimpleNamespace(merge_and_unload=lambda: log.append(("merge",)) or _FakeMerged(log))

    class WhisperForConditionalGeneration:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            log.append(("load_base", model_id, kwargs))
            return f"base<{model_id}>"

    class WhisperProcessor:
        @staticmethod
        def from_pretrained(source):
            log.append(("load_processor", source))
            return types.SimpleNamespace(
                save_pretrained=lambda path: log.append(("save_processor", path)),
                feature_extractor=types.SimpleNamespace(
                    save_pretrained=lambda path: log.append(("save_feature_extractor", path))),
            )

    class GenerationConfig:
        @staticmethod
        def from_pretrained(path):
            log.append(("load_generation_config", path))
            return "adapter-generation-config"

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=PeftModel))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        GenerationConfig=GenerationConfig,
        WhisperForConditionalGeneration=WhisperForConditionalGeneration,
        WhisperProcessor=WhisperProcessor,
    ))
    monkeypatch.setattr(compat, "fp32_load_kwargs", lambda: {"dtype": "float32"})
    return log


def test_merge_loads_base_from_the_adapter_config_applies_and_merges(tmp_path, fake_libs):
    adapter = _model_dir(tmp_path, adapter=True, base="org/base-whisper")
    (adapter / "preprocessor_config.json").write_text("{}")
    merged = tmp_path / "merged"
    merged.mkdir()

    assert export_ct2._merge_adapter(adapter, merged) == merged

    assert [step[0] for step in fake_libs] == [
        "load_base", "apply_adapter", "merge", "save_model",
        "load_processor", "save_processor", "save_feature_extractor",
    ]
    assert fake_libs[-1] == ("save_feature_extractor", str(merged))
    assert fake_libs[0] == ("load_base", "org/base-whisper", {"dtype": "float32"})
    assert fake_libs[1] == ("apply_adapter", "base<org/base-whisper>", str(adapter))
    assert (merged / "model.safetensors").exists()


def test_merge_takes_the_processor_from_the_adapter_dir_when_present_else_the_base(tmp_path, fake_libs):
    adapter = _model_dir(tmp_path, adapter=True)
    merged = tmp_path / "merged"
    merged.mkdir()

    export_ct2._merge_adapter(adapter, merged)
    assert ("load_processor", "org/base-whisper") in fake_libs

    fake_libs.clear()
    (adapter / "preprocessor_config.json").write_text("{}")
    export_ct2._merge_adapter(adapter, merged)
    assert ("load_processor", str(adapter)) in fake_libs


def test_merge_restores_the_generation_config_saved_next_to_the_adapter(tmp_path, fake_libs):
    adapter = _model_dir(tmp_path, adapter=True)
    merged = tmp_path / "merged"
    merged.mkdir()

    export_ct2._merge_adapter(adapter, merged)
    assert ("save_model", str(merged), "base-generation-config") in fake_libs

    fake_libs.clear()
    (adapter / "generation_config.json").write_text("{}")
    export_ct2._merge_adapter(adapter, merged)
    assert ("save_model", str(merged), "adapter-generation-config") in fake_libs


def test_merge_requires_the_adapter_to_name_its_base_model(tmp_path, fake_libs):
    adapter = _model_dir(tmp_path, adapter=True, base=None)
    with pytest.raises(ValueError, match="base model"):
        export_ct2._merge_adapter(adapter, tmp_path / "merged")
    assert fake_libs == []
