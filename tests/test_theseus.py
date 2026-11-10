"""Theseus: álgebra de Procrustes, rotación plantada (debe recuperar el delta exacto y la función), formas con
anchuras distintas, controles y aplicación del delta."""
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
from safetensors.torch import save_file  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from model_maxxing.transfer.apm import sha256_file  # noqa: E402
from model_maxxing.transfer.theseus import (LINEARS, align_and_transport, matched_noise,  # noqa: E402
                                            materialize_affine, procrustes, tensor_name, transport)

KW = dict(num_hidden_layers=2, hidden_size=16, num_attention_heads=4, num_key_value_heads=2, intermediate_size=24,
          vocab_size=64, tie_word_embeddings=False, max_position_embeddings=128,
          initializer_range=0.3)       # atención no uniforme: con 0,02 las cabezas de un grupo GQA son redundantes


def _orth(n, seed):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q.float()


def test_procrustes_recovers_square_and_semi_orthogonal():
    H = torch.randn(400, 12, generator=torch.Generator().manual_seed(0))
    Q = _orth(12, 1)
    assert torch.allclose(procrustes(H.T @ (H @ Q)), Q, atol=1e-4)
    # Rectangular (A ancho 12 → B estrecho 7): UVᵀ maximiza tr(TᵀC); recupera Qs exacto si H_A es isotrópica.
    Hw, _ = torch.linalg.qr(H.double())
    Hw = Hw.float() * 20
    Qs = Q[:, :7]
    T = procrustes(Hw.T @ (Hw @ Qs))
    assert torch.allclose(T, Qs, atol=1e-4) and torch.allclose(T.T @ T, torch.eye(7), atol=1e-5)
    # con H anisótropa NO coincide con el mínimo de mínimos cuadrados (salvedad documentada del método)
    assert not torch.allclose(procrustes(H.T @ (H @ Qs)), Qs, atol=1e-2)


def test_transport_formula_matches_definition():
    tau, Ti, To = torch.randn(10, 12), _orth(12, 2)[:, :5], _orth(10, 3)[:, :4]
    x_b = torch.randn(3, 5)
    # Δy_B = x_B T_inᵀ τᵀ T_out  ==  x_B τ_Bᵀ
    assert torch.allclose(x_b @ transport(tau, Ti, To).T, x_b @ Ti.T @ tau.T @ To, atol=1e-5)


def test_matched_noise_norm_and_determinism():
    tau = torch.randn(6, 4) * 3
    a, b = matched_noise(tau, "x", 0), matched_noise(tau, "x", 0)
    assert torch.equal(a, b) and torch.allclose(a.norm(), tau.norm()) and not torch.equal(a, matched_noise(tau, "x", 1))


def _save(model, d):
    model.save_pretrained(str(d))
    return d


def _tiny(seed):
    torch.manual_seed(seed)
    cfg = Qwen2Config(**KW)
    cfg._attn_implementation = "eager"
    m = Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.fill_(1.0)                     # normas = 1: la rotación del residual conmuta con RMSNorm
            elif n.endswith("bias"):
                p.add_(torch.randn_like(p) * 0.1)
    return m


def _rotated(m, Q, P):
    """B ≡ A con el residual rotado por Q (ortogonal) y las neuronas del MLP permutadas por P. Función idéntica."""
    b = _tiny(0)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    for k, v in sd.items():
        if k in ("model.embed_tokens.weight", "lm_head.weight"):
            sd[k] = v @ Q
        elif any(f".{x}.weight" in k for x in ("q_proj", "k_proj", "v_proj")):
            sd[k] = v @ Q
        elif "o_proj.weight" in k:
            sd[k] = Q.T @ v
        elif "gate_proj.weight" in k or "up_proj.weight" in k:
            sd[k] = (P @ v) @ Q
        elif "down_proj.weight" in k:
            sd[k] = Q.T @ v @ P.T
    b.load_state_dict(sd)
    return b


def _tokens(n, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 64, (T,), generator=g).tolist() for _ in range(n)]


def test_planted_rotation_recovers_delta_and_function(tmp_path):
    A = _tiny(1)
    Q, P = _orth(16, 7), torch.eye(24)[torch.randperm(24, generator=torch.Generator().manual_seed(3))]
    Aft = _tiny(1)
    with torch.no_grad():                                 # ajuste fino sintético: solo capas lineales
        for r in range(2):
            for lin in LINEARS:
                for kind in ("weight", "bias") if lin in ("q_proj", "k_proj", "v_proj") else ("weight",):
                    p = Aft.get_parameter(tensor_name(r, lin, kind))
                    p.add_(torch.randn_like(p) * 0.05)
    B, Bft_truth = _rotated(A, Q, P), _rotated(Aft, Q, P)
    da, dft, db = _save(A, tmp_path / "A"), _save(Aft, tmp_path / "Aft"), _save(B, tmp_path / "B")
    align_and_transport(da, dft, db, _tokens(12, 30), tmp_path / "deltas", "cpu", log=lambda *_: None)
    truth = Bft_truth.state_dict()
    Bsd = B.state_dict()
    for r in range(2):
        for lin in LINEARS:
            n = tensor_name(r, lin)
            from safetensors import safe_open
            with safe_open(str(tmp_path / "deltas" / "TH" / f"layer_{r:02d}.safetensors"), "pt") as f:
                got = f.get_tensor(n)
            assert torch.allclose(got, truth[n] - Bsd[n], atol=2e-5), n
    m = Qwen2ForCausalLM.from_pretrained(str(db), dtype=torch.float32).eval()
    materialize_affine(m, str(db), str(tmp_path / "deltas" / "TH"), alpha=1.0)
    ids = torch.tensor([_tokens(1, 20, seed=9)[0]])
    with torch.no_grad():
        assert torch.allclose(m(ids).logits, Aft(ids).logits, atol=1e-4)


def test_cross_width_shapes_controls_and_manifests(tmp_path):
    wide = dict(KW, hidden_size=32, num_attention_heads=8, num_key_value_heads=4, intermediate_size=48)
    torch.manual_seed(0)
    A = Qwen2ForCausalLM(Qwen2Config(**wide)).eval()
    Aft = Qwen2ForCausalLM(Qwen2Config(**wide)).eval()
    B = Qwen2ForCausalLM(Qwen2Config(**dict(KW, tie_word_embeddings=True))).eval()
    da, dft, db = _save(A, tmp_path / "A"), _save(Aft, tmp_path / "Aft"), _save(B, tmp_path / "B")
    before = {p.name: sha256_file(p) for d in (da, dft, db) for p in d.iterdir()}
    res = align_and_transport(da, dft, db, _tokens(10, 25), tmp_path / "d", "cpu", log=lambda *_: None)
    assert set(res["manifests"]) == {"TH", "RAND", "NAIVE"}
    for arm in ("TH", "RAND", "NAIVE"):
        m = Qwen2ForCausalLM.from_pretrained(str(db), dtype=torch.float32)
        info = materialize_affine(m, str(db), str(tmp_path / "d" / arm), alpha=0.5)
        assert info["delta_tensors_applied"] == 2 * (7 + 3)
        assert torch.isfinite(m(torch.tensor([[1, 2, 3]])).logits).all()
    diag = json.loads((tmp_path / "d" / "alignment_diagnostics.json").read_text())
    assert all(0 <= v <= 1.0001 for L in diag["layers"] for v in L["procrustes_similarity"].values())
    m = Qwen2ForCausalLM.from_pretrained(str(db), dtype=torch.float32)
    info = materialize_affine(m, str(db), str(tmp_path / "d" / "TH"), alpha=0.5,
                              exclude=("self_attn.q_proj.", "self_attn.k_proj."))
    assert info["delta_tensors_applied"] == 2 * (5 + 1)               # sin q/k (pesos y sesgos)
    with pytest.raises(FileExistsError):
        align_and_transport(da, dft, db, _tokens(10, 25), tmp_path / "d", "cpu", log=lambda *_: None)
    assert before == {p.name: sha256_file(p) for d in (da, dft, db) for p in d.iterdir()}


def test_materialize_affine_identity_shrink_and_apm_equivalence(tmp_path):
    B = _tiny(4)
    db = _save(B, tmp_path / "B")
    S = {k: torch.randn_like(v) for k, v in B.state_dict().items() if k != "lm_head.weight"}
    (tmp_path / "S").mkdir()
    save_file(S, str(tmp_path / "S" / "model.safetensors"))
    R = {k: v.clone() for k, v in B.state_dict().items()}
    m = Qwen2ForCausalLM.from_pretrained(str(db), dtype=torch.float32)
    materialize_affine(m, str(db))
    assert all(torch.equal(m.state_dict()[k], R[k]) for k in R)
    materialize_affine(m, str(db), scale=0.98)
    assert torch.allclose(m.state_dict()["model.norm.weight"], 0.98 * R["model.norm.weight"])
    materialize_affine(m, str(db), str(tmp_path / "S"), scale=0.98, alpha=0.02)       # = interpolación de APM
    k = "model.layers.0.mlp.down_proj.weight"
    assert torch.allclose(m.state_dict()[k], 0.98 * R[k] + 0.02 * S[k], atol=1e-6)


def test_procrustes_robust_to_massive_offsets():
    """Regresión: en Qwen (sesgos q/k enormes) la covarianza sin centrar tiene condicionamiento ~1e12; en float32
    la rotación salía arbitraria (0.5B real: error 1,1 en q_proj). Con float64 debe recuperar T = I si A = B."""
    from model_maxxing.transfer.theseus import procrustes_sv
    g = torch.Generator().manual_seed(0)
    H = torch.randn(3000, 12, generator=g, dtype=torch.float64)
    H[:, 3] += 3e5                                      # activación "masiva" casi constante
    C = H.T @ H
    S = torch.linalg.svdvals(C)
    assert S[0] / S[-1] > 1e9
    T, _, null = procrustes_sv(C)
    assert null == 0 and torch.allclose(T.double(), torch.eye(12, dtype=torch.float64), atol=1e-5)
