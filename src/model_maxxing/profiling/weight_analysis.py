"""Estadísticas de diferencias de pesos por tensor."""
from __future__ import annotations

import torch


def delta_stats(a: dict, b: dict) -> list[dict]:
    rows = []
    for k, x in a.items():
        if not x.is_floating_point():
            continue
        d = b[k].float() - x.float()
        xn = x.float().norm().item()
        rows.append({
            "tensor": k,
            "l2": d.norm().item(),
            "rel_l2": d.norm().item() / xn if xn else float("nan"),
            "max_abs": d.abs().max().item(),
            "cosine": torch.nn.functional.cosine_similarity(
                x.float().flatten(), b[k].float().flatten(), dim=0).item(),
        })
    return rows
