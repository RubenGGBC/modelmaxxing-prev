import pytest
import torch

from model_maxxing.surgery.patch_builder import apply_patch
from model_maxxing.surgery.patch_schema import BlockEdit, LowRankEdit, PatchSpec
from model_maxxing.surgery.patch_store import load_patch, patch_hash, save_patch
from model_maxxing.surgery.patch_validator import validate_patch
from model_maxxing.transfer.engine import survival, transfer

T = "model.layers.0.mlp.up_proj.weight"


def mk(seed):
    g = torch.Generator().manual_seed(seed)
    return {T: torch.randn(32, 32, generator=g), "model.layers.1.mlp.up_proj.weight": torch.randn(32, 32, generator=g)}


R, D = mk(1), mk(2)


def spec(scale=0.1, **kw):
    return PatchSpec(edits=(LowRankEdit(T, 4, 4, tuple([1.0] * 8), tuple([0.5] * 8), scale),), **kw)


def test_patch_is_nonzero_and_modifies():
    s = spec()
    v = validate_patch(s, R)
    assert v.ok and v.total_norm > 0
    Rs = apply_patch(R, s)
    assert not torch.equal(Rs[T], R[T])
    assert torch.equal(Rs["model.layers.1.mlp.up_proj.weight"], R["model.layers.1.mlp.up_proj.weight"])


def test_zero_patch_rejected():
    v = validate_patch(PatchSpec(edits=(BlockEdit(T, 0, 0, ((0.0, 0.0), (0.0, 0.0))),)), R)
    assert not v.ok and any("nulo" in e for e in v.errors)


def test_norm_bounds_and_allowlist_and_finite():
    assert not validate_patch(spec(scale=100.0), R).ok
    bad = PatchSpec(edits=(BlockEdit("model.embed_tokens.weight", 0, 0, ((1.0,),)),))
    assert not validate_patch(bad, {**R, "model.embed_tokens.weight": torch.zeros(4, 4)}).ok
    assert not validate_patch(PatchSpec(edits=(BlockEdit(T, 0, 0, ((float("nan"),),)),)), R).ok
    assert not validate_patch(PatchSpec(edits=(BlockEdit(T, 31, 31, ((1.0, 1.0),)),)), R).ok


def test_patch_persists_after_interpolation_and_vanishes_at_alpha1():
    s = spec(); Rs = apply_patch(R, s)
    sv = survival(transfer(Rs, D, "interpolate", 0.5), transfer(R, D, "interpolate", 0.5), [T])
    direct = (Rs[T] - R[T]).norm().item()
    assert sv[T] == pytest.approx(0.5 * direct, rel=1e-4)
    assert survival(transfer(Rs, D, "interpolate", 1.0), transfer(R, D, "interpolate", 1.0), [T])[T] < 1e-6


def test_rollback_and_originals_unmutated():
    before = {k: v.clone() for k, v in R.items()}
    Rs = apply_patch(R, spec())
    assert all(torch.equal(R[k], before[k]) for k in R)
    inverse = PatchSpec(edits=(LowRankEdit(T, 4, 4, tuple([1.0] * 8), tuple([0.5] * 8), -0.1),))
    assert torch.allclose(apply_patch(Rs, inverse)[T], R[T], atol=1e-6)


def test_store_roundtrip_hash(tmp_path):
    s = spec(justification="x")
    pid = save_patch(s, str(tmp_path))
    assert pid == patch_hash(s) and load_patch(pid, str(tmp_path)) == s


def test_transfer_baselines():
    assert all(torch.allclose(transfer(R, D, "interpolate", 0.0)[k], R[k]) for k in R)
    with pytest.raises(ValueError):
        transfer(R, D, "task_vector", 0.5)
