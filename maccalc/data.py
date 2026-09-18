"""Hardware and model tables shipped with the package."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


def chips() -> list[dict]:
    """Apple Silicon chips, M1 through M6, with bandwidth and RAM options."""
    return _load("chips.json")["apple_silicon"]


def accelerators() -> list[dict]:
    """Discrete GPUs and unified memory boxes, for cross vendor comparison."""
    return _load("chips.json")["accelerators"]


def models() -> list[dict]:
    """Model table with parameter counts and KV cache allowances."""
    return _load("models.json")["models"]


def quants() -> list[dict]:
    """Quantisation levels with bits per weight."""
    return _load("models.json")["quants"]


def benchmarks() -> dict:
    """The measured runs the speed model was fitted on, plus the method."""
    return _load("benchmarks.json")


def find_chip(query: str) -> dict | None:
    q = query.lower().replace(" ", "-")
    for c in chips():
        if c["id"] == q or c["name"].lower() == query.lower():
            return c
    return None


def find_model(query: str) -> dict | None:
    q = query.lower()
    for m in models():
        if m["slug"] == q or m["name"].lower() == q:
            return m
    return None


def bits_per_weight(quant_id: str) -> float:
    for q in quants():
        if q["id"] == quant_id or q["label"].lower() == quant_id.lower():
            return q["bpw"]
    raise KeyError(f"unknown quant: {quant_id}")
