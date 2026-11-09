from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from model_maxxing.surgery.patch_builder import edit_block, patch_tensors
from model_maxxing.surgery.patch_schema import PatchSpec


@dataclass
class PatchValidation:
    ok: bool
    errors: list[str]
    relative_norms: dict
    total_norm: float


def validate_patch(spec: PatchSpec, state: dict) -> PatchValidation:
    errs: list[str] = []
    if not spec.edits:
        errs.append("parche vacío")
    pat = re.compile(spec.allowed_pattern)
    for e in spec.edits:
        if e.tensor not in state:
            errs.append(f"tensor inexistente: {e.tensor}"); continue
        if not pat.fullmatch(e.tensor):
            errs.append(f"tensor fuera de la lista permitida: {e.tensor}"); continue
        blk = edit_block(e)
        if blk.ndim != 2 or blk.numel() == 0:
            errs.append(f"bloque no 2D/vacío en {e.tensor}"); continue
        if not torch.isfinite(blk).all():
            errs.append(f"NaN/Inf en {e.tensor}"); continue
        H, W = state[e.tensor].shape
        h, w = blk.shape
        if e.row0 < 0 or e.col0 < 0 or e.row0 + h > H or e.col0 + w > W:
            errs.append(f"bloque fuera de rango en {e.tensor}")
    rel, total = {}, 0.0
    if not errs:
        for k, d in patch_tensors(spec, state).items():
            n = d.norm().item()
            rel[k] = n / max(state[k].float().norm().item(), 1e-12)
            total += n * n
            if rel[k] > spec.max_relative_norm:
                errs.append(f"norma relativa {rel[k]:.4g} > {spec.max_relative_norm} en {k}")
        total = math.sqrt(total)
        if total == 0.0:
            errs.append("||P|| == 0 (parche nulo)")
    return PatchValidation(not errs, errs, rel, total)
