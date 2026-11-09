import pytest
import torch
from safetensors.torch import save_file

from model_maxxing.diagnostics.capability_gap import gap_by_subskill, tag_subskills
from model_maxxing.models.loading import Recipe, combine, dtype_report, materialize, upcast_modules


def _tiny(dtype=torch.bfloat16):
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16, bias=False), torch.nn.Linear(16, 4, bias=False)).to(dtype)


def _save(tmp_path, name, scale):
    torch.manual_seed({"R": 1, "D": 2, "B": 3}[name])
    sd = {"0.weight": torch.randn(16, 8) * scale, "1.weight": torch.randn(4, 16) * scale}
    d = tmp_path / name
    d.mkdir()
    save_file({k: v.to(torch.bfloat16) for k, v in sd.items()}, str(d / "model.safetensors"))
    return str(d), {k: v.to(torch.bfloat16).float() for k, v in sd.items()}


def test_combine_formulas():
    r, d, b = torch.ones(2), torch.full((2,), 3.0), torch.full((2,), 2.0)
    assert torch.equal(combine(r, Recipe("x"), None, None), r)
    assert torch.allclose(combine(r, Recipe("x", "task_vector", 0.5), d, b), torch.full((2,), 1.5))
    assert torch.allclose(combine(r, Recipe("x", "interpolate", 0.25), d, None), torch.full((2,), 1.5))


def test_upcast_keeps_outputs_and_dtypes():
    m = _tiny()
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    y0 = m(x).float()
    upcast_modules(m, ["1"])
    assert m[1].weight.dtype == torch.float32 and m[0].weight.dtype == torch.bfloat16
    y1 = m(x)
    assert y1.dtype == torch.bfloat16
    assert torch.allclose(y0, y1.float(), atol=0.05)
    rep = dtype_report(m)
    assert rep == {"torch.bfloat16": 128, "torch.float32": 64}


def test_upcast_preserves_small_patch_that_bf16_loses():
    m = _tiny()
    upcast_modules(m, ["1"])
    w = m[1].weight.data.clone()
    m[1].weight.data += 1e-4
    assert (m[1].weight.data - w).abs().max().item() > 5e-5          # FP32 conserva el parche
    lost = (w.to(torch.bfloat16) + 1e-4).float() - w.to(torch.bfloat16).float()
    assert lost.abs().max().item() == 0.0 or lost.abs().mean().item() < 1e-4


def test_materialize_task_vector_and_patch(tmp_path):
    rp, R = _save(tmp_path, "R", 1.0)
    dp, D = _save(tmp_path, "D", 1.0)
    bp, B = _save(tmp_path, "B", 1.0)
    m = _tiny()
    upcast_modules(m, ["1"])
    P = {"1.weight": torch.full((4, 16), 1e-3)}
    info = materialize(m, Recipe(rp, "task_vector", 0.5, dp, bp, patch=P))
    exp1 = R["1.weight"] + P["1.weight"] + 0.5 * (D["1.weight"] - B["1.weight"])
    assert torch.allclose(m[1].weight.data, exp1, atol=1e-6)            # tensor FP32: exacto
    exp0 = (R["0.weight"] + 0.5 * (D["0.weight"] - B["0.weight"])).to(torch.bfloat16)
    assert torch.equal(m[0].weight.data, exp0)
    assert abs(info["applied_patch_norm_after_transfer"] - P["1.weight"].norm().item()) < 1e-5
    # rollback: identidad devuelve exactamente R y no se modificó el checkpoint en disco
    materialize(m, Recipe(rp))
    assert torch.allclose(m[1].weight.data, R["1.weight"], atol=0)
    from safetensors.torch import load_file
    on_disk = load_file(rp + "/model.safetensors")
    assert torch.equal(on_disk["1.weight"].float(), R["1.weight"])


def test_materialize_rejects_unknown_patch_tensor(tmp_path):
    rp, _ = _save(tmp_path, "R", 1.0)
    with pytest.raises(KeyError):
        materialize(_tiny(), Recipe(rp, patch={"nope.weight": torch.zeros(1)}))


def test_subskill_tags():
    rec = "def f(n):\n    return 1 if n < 2 else n * f(n - 1)\n"
    assert "recursion" in tag_subskills(rec, "Write a function to compute factorial")
    s = "import re\ndef g(s):\n    return re.sub(' +', ' ', s)\n"
    assert {"regex", "string"} <= set(tag_subskills(s, "remove multiple spaces in a string"))
    assert "recursion" not in tag_subskills(s, "")
    ex = [{"task_id": 1, "code": rec, "prompt": "factorial"}, {"task_id": 2, "code": s, "prompt": "string"}]
    g = gap_by_subskill(ex, {1: False, 2: True}, {1: True, 2: True})
    assert g["recursion"] == {"n": 1, "R": 0, "D": 1, "gap_pp": 100.0, "R_fail_D_pass": 1, "R_pass_D_fail": 0}
    assert g["all"]["n"] == 2


def test_splits_disjoint_and_sizes():
    pytest.importorskip("datasets")
    from model_maxxing.evaluation.splits import check_disjoint, load_split
    try:
        parts = {k: load_split(k) for k in ("diagnostic", "selection", "final_mbpp")}
    except Exception as e:  # sin red ni caché
        pytest.skip(f"datasets no disponibles: {e}")
    assert [len(parts[k]) for k in parts] == [384, 333, 157]
    check_disjoint(parts)
