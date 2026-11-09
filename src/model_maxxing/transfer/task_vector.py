"""Aritmética de task vectors: R + a*(D - B). Solo válida si D deriva de B (ver compatibility)."""
from __future__ import annotations

import torch

from model_maxxing.models.compatibility import require_compatible
from model_maxxing.transfer.interpolation import check_finite


def compute_delta(donor: dict, base: dict) -> dict:
    require_compatible(donor, base)
    return {k: (donor[k].float() - base[k].float()) if donor[k].is_floating_point() else None
            for k in donor}


def apply_delta(receiver: dict, delta: dict, alpha: float, masks: dict | None = None) -> dict:
    require_compatible(receiver, {k: receiver[k] for k in delta})
    out = {}
    for k, r in receiver.items():
        dk = delta.get(k)
        if dk is None:
            out[k] = r.clone()
            continue
        if masks is not None and k in masks:
            dk = dk * masks[k].to(dk.dtype)
        out[k] = (r.float() + alpha * dk).to(r.dtype)
    check_finite(out)
    return out
