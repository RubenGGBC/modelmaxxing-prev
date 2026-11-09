import torch

from model_maxxing.surgery.arms import (make_spec, random_like, spec_from_matrix, total_norm, truncated, validate_budget)
from model_maxxing.surgery.patch_builder import apply_patch
from model_maxxing.surgery.patch_validator import validate_patch

T = "model.layers.3.mlp.down_proj.weight"
W = {T: torch.randn(24, 40)}


def test_decomposition_reconstructs_truncation_and_norm():
    M = torch.randn(24, 40)
    es = spec_from_matrix(T, M, 3, "x")
    spec = make_spec(es, "x", "")
    P = apply_patch({T: torch.zeros(24, 40)}, spec)[T]
    assert torch.allclose(P, truncated(M, 3), atol=1e-4)
    assert abs(total_norm(es) - truncated(M, 3).norm().item()) < 1e-3


def test_random_control_same_budget_different_direction():
    es = spec_from_matrix(T, 0.1 * torch.randn(24, 40), 3, "x")
    spec = make_spec(es, "x", "")
    ctl = random_like(spec, 0)
    assert abs(total_norm(list(ctl.edits)) - total_norm(list(spec.edits))) / total_norm(list(spec.edits)) < 0.5  # u,v aleatorios no ortogonales
    assert validate_budget(ctl, 1, 3, total_norm(list(ctl.edits))) == []
    assert validate_patch(ctl, W).ok


def test_budget_rejects_excess():
    es = spec_from_matrix(T, torch.randn(24, 40), 4, "x")
    spec = make_spec(es, "x", "")
    errs = validate_budget(spec, 1, 3, 0.1)
    assert any("rango" in e for e in errs) and any("norma" in e for e in errs)
