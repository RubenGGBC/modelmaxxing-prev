import pytest
import torch

from model_maxxing.models.compatibility import IncompatibleModels
from model_maxxing.transfer.interpolation import interpolate
from model_maxxing.transfer.masks import layer_range
from model_maxxing.transfer.task_vector import apply_delta, compute_delta


def mk(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "model.layers.0.mlp.w": torch.randn(4, 4, generator=g),
        "model.layers.1.mlp.w": torch.randn(4, 4, generator=g),
        "ids": torch.arange(3),
    }


R, D, B = mk(1), mk(2), mk(3)


def eq(a, b):
    return all(torch.allclose(a[k].float(), b[k].float(), atol=1e-6) for k in a)


def test_interpolation_identity():
    assert eq(interpolate(R, D, 0.0), R)
    assert eq(interpolate(R, D, 1.0), D if False else {**D, "ids": R["ids"]})


def test_delta_identity():
    assert eq(apply_delta(R, compute_delta(D, B), 0.0), R)


def test_empty_mask_is_noop():
    delta = compute_delta(D, B)
    masks = layer_range(R, 5, 6)  # ninguna capa
    assert eq(apply_delta(R, delta, 1.0, masks), R)


def test_incompatible_rejected():
    bad = dict(D)
    bad["model.layers.0.mlp.w"] = torch.zeros(2, 2)
    with pytest.raises(IncompatibleModels):
        interpolate(R, bad, 0.5)
    with pytest.raises(IncompatibleModels):
        interpolate(R, {"x": torch.zeros(1)}, 0.5)


def test_integrity_nan_rejected():
    bad = {**D, "model.layers.0.mlp.w": torch.full((4, 4), float("nan"))}
    with pytest.raises(ValueError):
        interpolate(R, bad, 0.5)


def test_reproducible():
    assert eq(interpolate(R, D, 0.3), interpolate(R, D, 0.3))
