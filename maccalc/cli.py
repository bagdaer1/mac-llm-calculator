"""Command line interface: maccalc <model> <chip> [options]."""

from __future__ import annotations

import argparse
import sys

from . import data, model as m


def _fmt(fit: m.Fit, model_name: str, chip_name: str, memory_gb: float, minimum_gb: float) -> str:
    verdict = "fits" if fit.fits else ("tight" if fit.tight else "does not fit")
    lines = [
        f"{model_name} on {chip_name} with {memory_gb:g} GB: {verdict}",
        f"  weights        {fit.weights_gb:g} GB",
        f"  context        {fit.context_gb:g} GB",
        f"  needs          {fit.comfortable_gb:g} GB including a {m.OS_RESERVE_GB:g} GB system reserve",
    ]
    if fit.tokens_per_second is not None:
        note = " (speed from active parameters, memory from total)" if fit.is_moe else ""
        lines.append(f"  generation     {fit.tokens_per_second:g} tokens per second{note}")
    else:
        lines.append(f"  needs a machine with at least {minimum_gb:g} GB at this quant, or drop to a smaller one")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="maccalc", description="Will this LLM run on this Mac, and how fast.")
    p.add_argument("model", help="model slug or name, or a parameter count like 70b")
    p.add_argument("chip", help="chip id or name, for example m4-pro, m5-ultra")
    p.add_argument("--ram", type=float, help="memory in GB, default is the largest option for the chip")
    p.add_argument("--quant", default="q4", help="q4, q5, q8 or fp16, default q4")
    p.add_argument("--ctx", type=int, default=8192, help="context window in tokens, default 8192")
    p.add_argument("--list-chips", action="store_true", help="print the chip table and exit")
    p.add_argument("--list-models", action="store_true", help="print the model table and exit")
    args = p.parse_args(argv)

    if args.list_chips:
        for c in data.chips():
            print(f"{c['id']:<12} {c['name']:<12} {c['bandwidth_gbs']:>5} GB/s  up to {max(c['ram_options_gb'])} GB")
        return 0
    if args.list_models:
        for x in data.models():
            print(f"{x['slug']:<18} {x['name']:<18} {x['params_b']}B")
        return 0

    chip = data.find_chip(args.chip)
    if chip is None:
        print(f"unknown chip: {args.chip}. Try --list-chips.", file=sys.stderr)
        return 2

    guide = data.find_model(args.model)
    if guide is None:
        raw = args.model.lower().rstrip("b")
        try:
            params_b = float(raw)
        except ValueError:
            print(f"unknown model: {args.model}. Try --list-models, or pass a size like 70b.", file=sys.stderr)
            return 2
        guide = {"name": f"{params_b:g}B model", "params_b": params_b, "ctx_overhead_gb": 2, "active_b": None}

    bpw = data.bits_per_weight(args.quant)
    ram = args.ram if args.ram else float(max(chip["ram_options_gb"]))
    fit = m.fit_model(
        params_b=guide["params_b"],
        bits_per_weight=bpw,
        memory_gb=ram,
        bandwidth_gbs=chip["bandwidth_gbs"],
        ctx_overhead_gb=guide.get("ctx_overhead_gb") or 2,
        ctx_tokens=args.ctx,
        active_params_b=guide.get("active_b"),
    )
    minimum = m.minimum_memory_gb(
        params_b=guide["params_b"],
        bits_per_weight=bpw,
        ctx_overhead_gb=guide.get("ctx_overhead_gb") or 2,
        ctx_tokens=args.ctx,
    )
    print(_fmt(fit, guide["name"], chip["name"], ram, minimum))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
