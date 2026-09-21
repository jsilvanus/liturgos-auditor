"""Cheap checks before a training run: will it fit in GPU memory?

They run before any weights are downloaded or loaded, so a run that cannot fit
fails in seconds with advice instead of after a multi-GB download and a CUDA
out-of-memory error. The numbers are deliberately rough; --skip-preflight
bypasses the check when the estimate is wrong for your setup.
"""

import logging

logger = logging.getLogger(__name__)

PRESETS = {
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "small": "openai/whisper-small",
}
DEFAULT_PRESET = "large-v3-turbo"

# Exact parameter counts of the preset checkpoints, so the check needs no download.
KNOWN_PARAMS = {
    "openai/whisper-large-v3-turbo": 808_878_080,
    "openai/whisper-small": 241_734_912,
}

GB = 1024 ** 3

# Assumptions behind estimate_training_memory_gb (bytes per parameter unless noted).
FULL_STATE_BYTES = 16          # fp32 weights + fp32 gradients + Adam first/second moments
FROZEN_WEIGHT_BYTES = 4        # LoRA keeps the frozen base in fp32
AUTOCAST_CACHE_BYTES = 2       # fp16/bf16 autocast caches half-precision copies of the weights
LORA_TRAINABLE_FRACTION = 0.02  # adapters on q/v are ~1%, on more modules a few percent
ACTIVATION_BYTES = 4           # per sample, half precision, no checkpointing
CHECKPOINTING_FACTOR = 0.15    # checkpointing keeps layer boundaries and recomputes the rest
FIXED_OVERHEAD_GB = 1.0        # CUDA context, cuBLAS workspaces, allocator fragmentation


class PreflightError(RuntimeError):
    """The requested run is not expected to fit the available hardware."""


def resolve_model_id(model_id=None, preset=None):
    """An explicit model id wins over a preset; with neither, the default preset applies."""
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose one of {', '.join(sorted(PRESETS))}")
    if model_id:
        if preset is not None and PRESETS[preset] != model_id:
            logger.warning("Model %s overrides preset %s", model_id, preset)
        return model_id
    return PRESETS[preset or DEFAULT_PRESET]


def estimate_training_memory_gb(num_params, mode, fp16, gradient_checkpointing, batch_size):
    """Rough peak GPU memory in GB for one training process.

    mode "full": fp32 weights + gradients + Adam states (16 bytes/param).
    mode "lora": frozen fp32 weights plus gradients and Adam states for the small adapters.
    fp16 stands for any 16-bit mixed precision (bf16 costs the same): it adds the
    autocast weight copies and halves activations. Activations are estimated
    per sample from the parameter count: Whisper always encodes a fixed 30 s window,
    so they scale with model depth and width rather than with the audio length.
    """
    if mode == "full":
        states = num_params * FULL_STATE_BYTES
    elif mode == "lora":
        states = num_params * FROZEN_WEIGHT_BYTES + num_params * LORA_TRAINABLE_FRACTION * FULL_STATE_BYTES
    else:
        raise ValueError(f"mode must be 'full' or 'lora', got {mode!r}")
    if fp16:
        states += num_params * AUTOCAST_CACHE_BYTES

    activations = num_params * ACTIVATION_BYTES * batch_size * (1 if fp16 else 2)
    if gradient_checkpointing:
        activations *= CHECKPOINTING_FACTOR
    return (states + activations) / GB + FIXED_OVERHEAD_GB


def count_parameters(model_id):
    """Parameter count of a Whisper checkpoint without loading its weights."""
    if model_id in KNOWN_PARAMS:
        return KNOWN_PARAMS[model_id]
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(config)
    return sum(p.numel() for p in model.parameters())


def _cuda_memory_gb():
    """(free, total) GB of the current CUDA device, or None when there is no GPU."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free / GB, total / GB


def check_gpu_memory(model_id, mode, fp16, gradient_checkpointing, batch_size):
    """Raise PreflightError when the run is estimated not to fit in free GPU memory.

    On CPU (no CUDA device) it only warns: training works but is extremely slow.
    Returns {"estimate_gb", "free_gb"}, or None when nothing could be checked.
    """
    memory = _cuda_memory_gb()
    if memory is None:
        logger.warning(
            "No CUDA GPU found: training on CPU works but is extremely slow "
            "(days for whisper-large-v3-turbo). Use --preset small or --peft lora for a test run."
        )
        return None
    free_gb, total_gb = memory

    try:
        num_params = count_parameters(model_id)
    except Exception as exc:  # unreadable config: let the model load report the real problem
        logger.warning("Could not read the model config for %s (%s); skipping the memory check", model_id, exc)
        return None

    estimate_gb = estimate_training_memory_gb(num_params, mode, fp16, gradient_checkpointing, batch_size)
    logger.info("Preflight: ~%.1f GB needed, %.1f of %.1f GB GPU memory free", estimate_gb, free_gb, total_gb)
    if estimate_gb > free_gb:
        raise PreflightError(_advice(
            model_id, num_params, mode, fp16, gradient_checkpointing, batch_size,
            estimate_gb, free_gb, total_gb,
        ))
    return {"estimate_gb": estimate_gb, "free_gb": free_gb}


def _advice(model_id, num_params, mode, fp16, gradient_checkpointing, batch_size, estimate_gb, free_gb, total_gb):
    def estimate(**changes):
        settings = {
            "num_params": num_params, "mode": mode, "fp16": fp16,
            "gradient_checkpointing": gradient_checkpointing, "batch_size": batch_size,
        }
        settings.update(changes)
        return estimate_training_memory_gb(**settings)

    options = []
    if mode == "full":
        options.append(f"--peft lora (~{estimate(mode='lora'):.1f} GB)")
    small = PRESETS["small"]
    if model_id != small and KNOWN_PARAMS[small] < num_params:
        options.append(f"--preset small (~{estimate(num_params=KNOWN_PARAMS[small]):.1f} GB)")
    if not gradient_checkpointing:
        options.append(f"--gradient-checkpointing (~{estimate(gradient_checkpointing=True):.1f} GB)")
    if not fp16:
        options.append(f"--fp16, or --bf16 on Ampere or newer GPUs (~{estimate(fp16=True):.1f} GB)")
    if batch_size > 1:
        options.append(
            f"a smaller --batch-size (~{estimate(batch_size=1):.1f} GB at 1; "
            "raise --gradient-accumulation-steps to keep the effective batch)"
        )
    options.append("a GPU with more memory")

    lines = [
        f"Training {model_id} ({'LoRA' if mode == 'lora' else 'full fine-tuning'}) needs about "
        f"{estimate_gb:.1f} GB of GPU memory, but only {free_gb:.1f} GB of {total_gb:.1f} GB is free.",
        "Options (estimated memory in parentheses, each applied on its own):",
    ]
    lines += [f"  - {option}" for option in options]
    together = estimate(mode="lora", fp16=True, gradient_checkpointing=True, batch_size=1)
    if together < estimate_gb:
        lines.append(f"All the memory savers combined (LoRA, 16-bit, checkpointing, batch size 1): ~{together:.1f} GB.")
    lines.append("Pass --skip-preflight to run anyway if you think the estimate is too high.")
    return "\n".join(lines)
