"""Máscaras simples por nombre de tensor (capas / módulos)."""
from __future__ import annotations

import re

import torch


def layer_index(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def module_mask(state: dict, include: callable) -> dict:
    """Máscara 1/0 (escalar expandible) por tensor según include(name)."""
    return {k: torch.full_like(v, 1.0 if include(k) else 0.0, dtype=torch.float32)
            for k, v in state.items() if v.is_floating_point()}


def layer_range(state: dict, lo: int, hi: int) -> dict:
    return module_mask(state, lambda k: (i := layer_index(k)) is not None and lo <= i < hi)


def by_kind(state: dict, kinds: tuple[str, ...]) -> dict:
    return module_mask(state, lambda k: any(s in k for s in kinds))
