"""Herramientas tipadas para el cirujano (LLM frontera). Sin acceso a código ni a escritura libre."""
from __future__ import annotations

import json

from model_maxxing.surgery.patch_schema import PatchSpec, spec_from_dict
from model_maxxing.surgery.patch_validator import validate_patch


def inspect_weight_region(state: dict, tensor: str, r0: int, r1: int, c0: int, c1: int, max_cells: int = 1024) -> dict:
    if (r1 - r0) * (c1 - c0) > max_cells:
        raise ValueError("región demasiado grande")
    w = state[tensor][r0:r1, c0:c1].float()
    return {"tensor": tensor, "slice": [r0, r1, c0, c1], "values": w.tolist(),
            "mean": w.mean().item(), "std": w.std().item(), "fro": w.norm().item()}


def inspect_delta_region(a: dict, b: dict, tensor: str, r0: int, r1: int, c0: int, c1: int, max_cells: int = 1024) -> dict:
    d = {tensor: (b[tensor].float() - a[tensor].float())}
    return inspect_weight_region(d, tensor, r0, r1, c0, c1, max_cells)


def parse_proposal(text: str, state: dict) -> tuple[PatchSpec, object]:
    """Convierte la propuesta JSON del LLM en PatchSpec validado; nunca ejecuta código."""
    spec = spec_from_dict(json.loads(text))
    return spec, validate_patch(spec, state)
