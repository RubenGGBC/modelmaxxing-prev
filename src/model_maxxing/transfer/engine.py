"""Operador de transferencia común para A/C/D/E y medida de supervivencia del parche."""
from __future__ import annotations

import torch

from model_maxxing.transfer.interpolation import interpolate
from model_maxxing.transfer.task_vector import apply_delta, compute_delta


def transfer(receiver: dict, donor: dict, mode: str, alpha: float, base: dict | None = None) -> dict:
    if mode == "interpolate":
        return interpolate(receiver, donor, alpha)
    if mode == "task_vector":
        if base is None:
            raise ValueError("task_vector requiere base verificada")
        return apply_delta(receiver, compute_delta(donor, base), alpha)
    raise ValueError(f"modo desconocido: {mode}")


def survival(final_with_patch: dict, final_without_patch: dict, tensors: list[str]) -> dict:
    """||W_final(P) - W_final(0)|| por tensor editado."""
    return {t: (final_with_patch[t].float() - final_without_patch[t].float()).norm().item() for t in tensors}
