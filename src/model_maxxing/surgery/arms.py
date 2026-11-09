"""Constructores de parches para los brazos B, C, E con presupuesto idéntico (v0.3).
Presupuesto: <= m tensores, rango <= k por tensor, norma Frobenius total <= N0.
El brazo D (Claude) emite su PatchSpec propio dentro del mismo presupuesto (ver validate_budget)."""
from __future__ import annotations

import math

import torch

from model_maxxing.surgery.patch_schema import LowRankEdit, PatchSpec

PAT = r"model\.layers\.\d+\.mlp\.down_proj\.weight"


def spec_from_matrix(tensor: str, M: torch.Tensor, k: int, author: str, just: str = "") -> list[LowRankEdit]:
    """Descompone M en sus k primeras componentes singulares como LowRankEdit (u,v de norma 1; scale=sigma)."""
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    return [LowRankEdit(tensor, 0, 0, tuple(U[:, i].tolist()), tuple(Vh[i].tolist()), float(S[i])) for i in range(k)]


def truncated(M: torch.Tensor, k: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vh[:k]


def rescale(edits: list[LowRankEdit], factor: float) -> list[LowRankEdit]:
    return [LowRankEdit(e.tensor, e.row0, e.col0, e.u, e.v, e.scale * factor) for e in edits]


def total_norm(edits: list[LowRankEdit]) -> float:
    """Norma Frobenius total (componentes ortonormales por tensor => sqrt(sum sigma^2) si u,v ortonormales;
    en general se calcula exacta por tensor)."""
    by = {}
    for e in edits:
        by.setdefault(e.tensor, []).append(e)
    tot = 0.0
    for t, es in by.items():
        U = torch.stack([torch.tensor(e.u) * e.scale for e in es], 1)
        V = torch.stack([torch.tensor(e.v) for e in es], 1)
        tot += (U @ V.T).pow(2).sum().item()
    return math.sqrt(tot)


def make_spec(edits: list[LowRankEdit], author: str, justification: str, rel: float = 0.5, meta: dict | None = None) -> PatchSpec:
    return PatchSpec(edits=tuple(edits), parent_id="Med-Receiver", justification=justification,
                     author=author, max_relative_norm=rel, allowed_pattern=PAT, meta=meta or {})


def validate_budget(spec: PatchSpec, m: int, k: int, n0: float, tol: float = 1e-3) -> list[str]:
    errs = []
    tensors = {}
    for e in spec.edits:
        tensors.setdefault(e.tensor, []).append(e)
    if len(tensors) > m:
        errs.append(f"{len(tensors)} tensores > m={m}")
    for t, es in tensors.items():
        if len(es) > k:
            errs.append(f"rango {len(es)} > k={k} en {t}")
    tn = total_norm(list(spec.edits))
    if tn > n0 * (1 + tol):
        errs.append(f"norma total {tn:.4f} > N0={n0:.4f}")
    return errs


def random_like(spec: PatchSpec, seed: int) -> PatchSpec:
    """Control E: mismos tensores, rango y escalas; direcciones aleatorias (misma norma por componente)."""
    g = torch.Generator().manual_seed(seed)
    edits = []
    for e in spec.edits:
        u = torch.randn(len(e.u), generator=g); u /= u.norm()
        v = torch.randn(len(e.v), generator=g); v /= v.norm()
        edits.append(LowRankEdit(e.tensor, e.row0, e.col0, tuple(u.tolist()), tuple(v.tolist()), e.scale))
    return PatchSpec(edits=tuple(edits), parent_id=spec.parent_id, author="control_random",
                     justification=f"control aleatorio de {spec.author}", max_relative_norm=spec.max_relative_norm,
                     allowed_pattern=spec.allowed_pattern, meta={"seed": seed, "mirrors": spec.author})
