"""Interpolación lineal: (1-a)*R + a*D. Mezcla en FP32, devuelve el dtype original."""
from __future__ import annotations

import torch

from model_maxxing.models.compatibility import require_compatible


def check_finite(state: dict) -> None:
    for k, v in state.items():
        if v.is_floating_point() and not torch.isfinite(v).all():
            raise ValueError(f"NaN/Inf en {k}")


def interpolate(receiver: dict, donor: dict, alpha: float) -> dict:
    require_compatible(receiver, donor)
    out = {}
    for k, r in receiver.items():
        if not r.is_floating_point():
            out[k] = r.clone()
            continue
        d = donor[k]
        out[k] = ((1 - alpha) * r.float() + alpha * d.float()).to(r.dtype)
    check_finite(out)
    return out
