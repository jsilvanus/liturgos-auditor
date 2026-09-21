import logging
import sys

import pytest

from auditor_stt.training import preflight
from auditor_stt.training.preflight import (
    GB,
    KNOWN_PARAMS,
    PRESETS,
    PreflightError,
    check_gpu_memory,
    estimate_training_memory_gb,
    resolve_model_id,
)

TURBO_ID = "openai/whisper-large-v3-turbo"
SMALL_ID = "openai/whisper-small"
TURBO = KNOWN_PARAMS[TURBO_ID]
SMALL = KNOWN_PARAMS[SMALL_ID]


def _estimate(num_params=TURBO, mode="full", fp16=True, gradient_checkpointing=False, batch_size=4):
    return estimate_training_memory_gb(num_params, mode, fp16, gradient_checkpointing, batch_size)


# --- resolve_model_id ---------------------------------------------------------

def test_default_resolution_is_large_v3_turbo():
    assert resolve_model_id() == TURBO_ID


def test_preset_selects_its_model():
    assert resolve_model_id(preset="small") == SMALL_ID
    assert resolve_model_id(preset="large-v3-turbo") == TURBO_ID


def test_explicit_model_wins_over_preset_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=preflight.logger.name):
        assert resolve_model_id("some/own-whisper", preset="small") == "some/own-whisper"
    assert "overrides preset small" in caplog.text


def test_explicit_model_matching_the_preset_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger=preflight.logger.name):
        assert resolve_model_id(SMALL_ID, preset="small") == SMALL_ID
    assert caplog.text == ""


def test_unknown_preset_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="large-v3-turbo, small"):
        resolve_model_id(preset="huge")


def test_presets_have_known_parameter_counts():
    assert set(PRESETS.values()) <= set(KNOWN_PARAMS)


# --- estimate_training_memory_gb ---------------------------------------------

def test_estimate_arithmetic_full_fp32_without_checkpointing():
    # 1e9 params: 16 B states + 4 B x 2 (fp32) x batch 1 activations, plus the fixed overhead.
    expected = (1e9 * 16 + 1e9 * 4 * 2) / GB + preflight.FIXED_OVERHEAD_GB
    assert _estimate(1e9, "full", fp16=False, batch_size=1) == pytest.approx(expected)


def test_estimate_arithmetic_lora_fp16_with_checkpointing():
    states = 1e9 * 4 + 1e9 * preflight.LORA_TRAINABLE_FRACTION * 16 + 1e9 * 2
    activations = 1e9 * 4 * 2 * preflight.CHECKPOINTING_FACTOR
    expected = (states + activations) / GB + preflight.FIXED_OVERHEAD_GB
    assert _estimate(1e9, "lora", fp16=True, gradient_checkpointing=True, batch_size=2) == pytest.approx(expected)


def test_full_finetune_of_turbo_needs_at_least_weights_grads_and_adam():
    # The plan's "roughly 13+ GB": 809M x 16 bytes.
    assert _estimate(mode="full", fp16=True, gradient_checkpointing=True, batch_size=1) > 13


def test_lora_saves_the_gradient_and_optimizer_state_of_the_frozen_weights():
    # Activations are assumed equal, so compare where weights and optimizer states dominate.
    full = _estimate(mode="full", gradient_checkpointing=True, batch_size=1)
    lora = _estimate(mode="lora", gradient_checkpointing=True, batch_size=1)
    assert lora < full / 2
    assert full - lora > 8  # GB: turbo's gradients and Adam states


def test_checkpointing_fp16_smaller_batch_and_smaller_model_each_reduce_the_estimate():
    base = _estimate(mode="full", fp16=False, gradient_checkpointing=False, batch_size=4)
    assert _estimate(mode="full", fp16=False, gradient_checkpointing=True, batch_size=4) < base
    assert _estimate(mode="full", fp16=True, gradient_checkpointing=False, batch_size=4) < base
    assert _estimate(mode="full", fp16=False, gradient_checkpointing=False, batch_size=1) < base
    assert _estimate(SMALL, mode="full", fp16=False, gradient_checkpointing=False, batch_size=4) < base


def test_turbo_lora_with_checkpointing_fits_a_free_8gb_card_but_not_full_finetuning():
    assert _estimate(mode="lora", fp16=True, gradient_checkpointing=True, batch_size=2) < 8
    assert _estimate(mode="full", fp16=True, gradient_checkpointing=True, batch_size=1) > 8


def test_estimate_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        _estimate(mode="qlora")


# --- check_gpu_memory ---------------------------------------------------------

def _gpu(monkeypatch, free_gb, total_gb=8.0):
    monkeypatch.setattr(preflight, "_cuda_memory_gb", lambda: (free_gb, total_gb))


def test_check_raises_actionable_error_when_full_finetune_cannot_fit(monkeypatch):
    _gpu(monkeypatch, free_gb=6.0)
    with pytest.raises(PreflightError) as excinfo:
        check_gpu_memory(TURBO_ID, "full", fp16=False, gradient_checkpointing=False, batch_size=4)
    message = str(excinfo.value)
    assert "6.0 GB" in message and "8.0 GB" in message
    for advice in ("--peft lora", "--preset small", "--gradient-checkpointing", "--fp16", "--batch-size",
                   "--gradient-accumulation-steps", "more memory", "--skip-preflight"):
        assert advice in message
    combined = _estimate(mode="lora", fp16=True, gradient_checkpointing=True, batch_size=1)
    assert f"combined (LoRA, 16-bit, checkpointing, batch size 1): ~{combined:.1f} GB" in message


def test_advice_omits_what_is_already_applied(monkeypatch):
    _gpu(monkeypatch, free_gb=1.0)
    with pytest.raises(PreflightError) as excinfo:
        check_gpu_memory(SMALL_ID, "lora", fp16=True, gradient_checkpointing=True, batch_size=1)
    message = str(excinfo.value)
    for applied in ("--peft lora", "--preset small", "--gradient-checkpointing", "--fp16", "--batch-size"):
        assert applied not in message
    assert "more memory" in message
    assert "combined" not in message  # nothing left to combine


def test_check_passes_and_reports_numbers_when_the_run_fits(monkeypatch):
    _gpu(monkeypatch, free_gb=7.5)
    result = check_gpu_memory(TURBO_ID, "lora", fp16=True, gradient_checkpointing=True, batch_size=1)
    assert result["free_gb"] == 7.5
    assert 0 < result["estimate_gb"] < 7.5


def test_check_only_warns_without_a_gpu(monkeypatch, caplog):
    monkeypatch.setattr(preflight, "_cuda_memory_gb", lambda: None)
    with caplog.at_level(logging.WARNING, logger=preflight.logger.name):
        result = check_gpu_memory(TURBO_ID, "full", fp16=False, gradient_checkpointing=False, batch_size=64)
    assert result is None
    assert "CPU" in caplog.text and "slow" in caplog.text


def test_check_skips_with_a_warning_when_the_model_config_is_unreadable(monkeypatch, caplog):
    _gpu(monkeypatch, free_gb=1.0)

    def unreadable(model_id):
        raise OSError("offline")

    monkeypatch.setattr(preflight, "count_parameters", unreadable)
    with caplog.at_level(logging.WARNING, logger=preflight.logger.name):
        assert check_gpu_memory("some/own-whisper", "full", True, False, 4) is None
    assert "skipping the memory check" in caplog.text


def test_check_uses_known_counts_without_touching_the_hub(monkeypatch):
    _gpu(monkeypatch, free_gb=100.0)
    monkeypatch.setitem(sys.modules, "transformers", None)  # any import of it would fail
    assert check_gpu_memory(TURBO_ID, "full", True, False, 4)["estimate_gb"] > 13
