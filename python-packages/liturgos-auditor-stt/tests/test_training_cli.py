import sys
import types

import pytest

from auditor_stt import cli
from auditor_stt.training import export_ct2


@pytest.fixture
def train_call(monkeypatch):
    """Stub `auditor_stt.training.train` so no torch is needed; returns the kwargs train() was called with."""
    seen = {}
    fake = types.ModuleType("auditor_stt.training.train")
    fake.train = lambda **kwargs: seen.update(kwargs)
    monkeypatch.setitem(sys.modules, "auditor_stt.training.train", fake)
    return seen


@pytest.fixture
def export_call(monkeypatch):
    seen = {}
    monkeypatch.setattr(export_ct2, "export", lambda **kwargs: seen.update(kwargs))
    return seen


def test_train_defaults_let_a_preset_apply_and_keep_full_finetuning(train_call):
    assert cli.main(["train", "--dataset", "data", "--out", "run"]) == 0

    assert train_call["dataset_dir"] == "data" and train_call["output_dir"] == "run"
    assert train_call["model_id"] is None  # so resolve_model_id can fall back to the preset default
    assert train_call["preset"] is None
    assert train_call["peft"] == "none"
    assert (train_call["lora_r"], train_call["lora_alpha"], train_call["lora_dropout"]) == (32, 64, 0.05)
    assert train_call["lora_target_modules"] == ["q_proj", "v_proj"]
    assert train_call["gradient_checkpointing"] is False
    assert train_call["skip_preflight"] is False
    assert train_call["fp16"] is False and train_call["bf16"] is False


def test_train_parses_all_new_flags(train_call):
    cli.main([
        "train", "--dataset", "data", "--out", "run",
        "--preset", "small", "--peft", "lora",
        "--lora-r", "8", "--lora-alpha", "16", "--lora-dropout", "0.1",
        "--lora-target-modules", "q_proj", "k_proj", "v_proj", "out_proj",
        "--gradient-checkpointing", "--skip-preflight", "--fp16",
        "--batch-size", "2", "--gradient-accumulation-steps", "8",
    ])

    assert train_call["preset"] == "small"
    assert train_call["peft"] == "lora"
    assert (train_call["lora_r"], train_call["lora_alpha"], train_call["lora_dropout"]) == (8, 16, 0.1)
    assert train_call["lora_target_modules"] == ["q_proj", "k_proj", "v_proj", "out_proj"]
    assert train_call["gradient_checkpointing"] is True
    assert train_call["skip_preflight"] is True
    assert train_call["fp16"] is True
    assert (train_call["batch_size"], train_call["gradient_accumulation_steps"]) == (2, 8)


def test_train_explicit_model_is_passed_alongside_the_preset(train_call):
    cli.main(["train", "--dataset", "d", "--out", "o", "--model", "org/own", "--preset", "small"])
    assert train_call["model_id"] == "org/own" and train_call["preset"] == "small"


@pytest.mark.parametrize("bad", [["--preset", "huge"], ["--peft", "qlora"], ["--lora-r", "many"]])
def test_train_rejects_invalid_values(train_call, bad, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["train", "--dataset", "d", "--out", "o", *bad])
    assert excinfo.value.code == 2
    assert train_call == {}


def test_train_preflight_refusal_is_an_error_message_not_a_traceback(monkeypatch, capsys):
    from auditor_stt.training.preflight import PreflightError

    def refuse(**kwargs):
        raise PreflightError("Training x needs about 37.2 GB.\nOptions:\n  - --peft lora")

    fake = types.ModuleType("auditor_stt.training.train")
    fake.train = refuse
    monkeypatch.setitem(sys.modules, "auditor_stt.training.train", fake)

    assert cli.main(["train", "--dataset", "d", "--out", "o"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: Training x needs about 37.2 GB.") and "--peft lora" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "failure",
    [ValueError("m is a LoRA adapter; pass --merge-lora"), FileExistsError("out already exists"), RuntimeError("no converter")],
)
def test_export_expected_failures_are_error_messages_not_tracebacks(monkeypatch, capsys, failure):
    def refuse(**kwargs):
        raise failure

    monkeypatch.setattr(export_ct2, "export", refuse)

    assert cli.main(["export", "--model", "run", "--out", "ct2"]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"error: {failure}") and "Traceback" not in err


def test_export_merge_lora_flag(export_call):
    cli.main(["export", "--model", "run", "--out", "ct2", "--merge-lora"])
    assert export_call == {"model_dir": "run", "output_dir": "ct2", "quantization": "float16", "merge_lora": True}


def test_export_does_not_merge_by_default(export_call):
    cli.main(["export", "--model", "run", "--out", "ct2", "--quantization", "int8"])
    assert export_call["merge_lora"] is False and export_call["quantization"] == "int8"
