"""Materializa un PatchSpec: delta denso solo por bloque editado; no muta el estado original."""
from __future__ import annotations

import torch

from model_maxxing.surgery.patch_schema import BlockEdit, LowRankEdit, PatchSpec


def edit_block(e) -> torch.Tensor:
    if isinstance(e, BlockEdit):
        return torch.tensor(e.values, dtype=torch.float32)
    return e.scale * torch.outer(torch.tensor(e.u, dtype=torch.float32), torch.tensor(e.v, dtype=torch.float32))


def patch_tensors(spec: PatchSpec, state: dict) -> dict:
    """Devuelve {tensor: delta denso FP32 del tamaño del tensor} solo para tensores tocados."""
    out: dict[str, torch.Tensor] = {}
    for e in spec.edits:
        blk = edit_block(e)
        if e.tensor not in out:
            out[e.tensor] = torch.zeros(state[e.tensor].shape, dtype=torch.float32)
        h, w = blk.shape
        out[e.tensor][e.row0:e.row0 + h, e.col0:e.col0 + w] += blk
    return out


def apply_patch(state: dict, spec: PatchSpec) -> dict:
    """R* = R + P. Devuelve un dict nuevo; solo se clonan los tensores tocados."""
    deltas = patch_tensors(spec, state)
    new = dict(state)
    for k, d in deltas.items():
        new[k] = (state[k].float() + d).to(state[k].dtype)
    return new
