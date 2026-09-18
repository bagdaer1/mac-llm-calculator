"""Memory and speed model for local LLM inference.

Two numbers decide whether a model runs on a machine and how fast:
how much memory the weights plus context need, and how many gigabytes
the runtime reads per generated token.

Speed model, fitted by least squares to six measured runs on a base M4
(see data/benchmarks.json):

    seconds_per_token = read_gb / (bandwidth_gbs * 0.915) + 0.003885

The first term is the memory bus. The second is fixed per token overhead
that does not scale with model size. A single term model overestimates
small models badly: without the overhead term the same formula claims an
RTX 5090 generates over 300 tokens per second on an 8B model, which
nobody observes.
"""

from __future__ import annotations

from dataclasses import dataclass

BANDWIDTH_EFFICIENCY = 0.915
"""Share of rated memory bandwidth reached during generation."""

PER_TOKEN_OVERHEAD_S = 0.003885
"""Fixed cost per token, seconds. Fitted on Apple Silicon with Ollama."""

RUNTIME_OVERHEAD = 1.15
"""Runtime allocates about 15% above the raw weight size."""

OS_RESERVE_GB = 3.0
"""Memory left to macOS. Use 1 GB for a discrete GPU with its own VRAM."""


def weights_gb(params_b: float, bits_per_weight: float) -> float:
    """Weight size in GB for a parameter count in billions."""
    return params_b * bits_per_weight / 8


def context_gb(ctx_overhead_gb: float, ctx_tokens: int) -> float:
    """KV cache allowance, scaled from the per model figure at 32K tokens."""
    return max(1.0, round(ctx_overhead_gb * (ctx_tokens / 32768) * 10) / 10)


def tokens_per_second(bandwidth_gbs: float, read_gb: float) -> float:
    """Generation speed for a machine that reads read_gb per token."""
    if bandwidth_gbs <= 0 or read_gb <= 0:
        raise ValueError("bandwidth and read size must be positive")
    seconds = read_gb / (bandwidth_gbs * BANDWIDTH_EFFICIENCY) + PER_TOKEN_OVERHEAD_S
    raw = 1 / seconds
    return round(raw, 1) if raw < 50 else float(round(raw))


@dataclass
class Fit:
    weights_gb: float
    context_gb: float
    required_gb: float
    """Weights plus runtime overhead plus context, without the OS reserve."""
    comfortable_gb: float
    """What the machine must actually have."""
    fits: bool
    tight: bool
    """Fits only by eating into the OS reserve."""
    tokens_per_second: float | None
    is_moe: bool


def fit_model(
    params_b: float,
    bits_per_weight: float,
    memory_gb: float,
    bandwidth_gbs: float,
    ctx_overhead_gb: float = 2.0,
    ctx_tokens: int = 8192,
    active_params_b: float | None = None,
    os_reserve_gb: float = OS_RESERVE_GB,
) -> Fit:
    """Does this model fit on this machine, and how fast will it generate.

    For a mixture of experts model pass active_params_b: memory comes from
    the total parameter count, speed from the active one.
    """
    w = weights_gb(params_b, bits_per_weight)
    c = context_gb(ctx_overhead_gb, ctx_tokens)
    required = w * RUNTIME_OVERHEAD + c
    comfortable = required + os_reserve_gb
    fits = comfortable <= memory_gb
    tight = not fits and required <= memory_gb
    read = weights_gb(active_params_b, bits_per_weight) if active_params_b else w
    speed = tokens_per_second(bandwidth_gbs, read) if (fits or tight) else None
    return Fit(
        weights_gb=round(w, 2),
        context_gb=c,
        required_gb=round(required, 1),
        comfortable_gb=round(comfortable, 1),
        fits=fits,
        tight=tight,
        tokens_per_second=speed,
        is_moe=active_params_b is not None,
    )


def minimum_memory_gb(
    params_b: float,
    bits_per_weight: float,
    ctx_overhead_gb: float = 2.0,
    ctx_tokens: int = 8192,
    os_reserve_gb: float = OS_RESERVE_GB,
) -> float:
    """Smallest memory configuration that runs this model comfortably."""
    w = weights_gb(params_b, bits_per_weight)
    return round(w * RUNTIME_OVERHEAD + context_gb(ctx_overhead_gb, ctx_tokens) + os_reserve_gb, 1)
