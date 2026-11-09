"""Métricas por problema y bootstrap pareado."""
from __future__ import annotations

import numpy as np


def accuracy(correct: list[bool]) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def paired_bootstrap(a: list[bool], b: list[bool], n: int = 10000, seed: int = 42, ci: float = 0.95) -> dict:
    """Diferencia media (b - a) con IC por remuestreo pareado de problemas."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("Resultados no alineados por problema")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {
        "diff": float((b - a).mean()), "ci_low": float(lo), "ci_high": float(hi),
        "wrong_to_right": int(((a == 0) & (b == 1)).sum()),
        "right_to_wrong": int(((a == 1) & (b == 0)).sum()),
    }
