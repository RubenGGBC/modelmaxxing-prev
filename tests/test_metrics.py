import pytest

from model_maxxing.evaluation.metrics import paired_bootstrap


def test_bootstrap_counts_and_diff():
    r = paired_bootstrap([1, 0, 0, 1], [1, 1, 0, 0], n=500)
    assert r["wrong_to_right"] == 1 and r["right_to_wrong"] == 1
    assert r["diff"] == 0 and r["ci_low"] <= 0 <= r["ci_high"]


def test_bootstrap_misaligned():
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 0])
