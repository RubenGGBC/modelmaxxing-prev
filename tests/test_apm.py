"""APM con modelos Qwen2 sintéticos diminutos (misma topología que 7B→1.5B: donante con lm_head propio,
receptor con embeddings atados, GQA con grupos de distinto tamaño)."""
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from model_maxxing.models.loading import materialize  # noqa: E402
from model_maxxing.transfer.apm import (ActivationStats, ArchSpec, apm_recipe, build_slice,  # noqa: E402
                                        check_compat, random_selection, select_apm, select_im, sha256_file)
from model_maxxing.transfer.apm_profile import profile_donor  # noqa: E402

DONOR = dict(num_hidden_layers=3, hidden_size=48, num_attention_heads=6, num_key_value_heads=2,
             intermediate_size=96, vocab_size=128, tie_word_embeddings=False)
RECIP = dict(num_hidden_layers=2, hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
             intermediate_size=64, vocab_size=120, tie_word_embeddings=True)
QWEN_7B = dict(model_type="qwen2", num_hidden_layers=28, hidden_size=3584, num_attention_heads=28,
               num_key_value_heads=4, intermediate_size=18944, vocab_size=152064, tie_word_embeddings=False)
QWEN_15B = dict(model_type="qwen2", num_hidden_layers=28, hidden_size=1536, num_attention_heads=12,
                num_key_value_heads=2, intermediate_size=8960, vocab_size=151936, tie_word_embeddings=True)


def _save(tmp, name, kw, seed):
    torch.manual_seed(seed)
    cfg = Qwen2Config(max_position_embeddings=64, **kw)
    cfg._attn_implementation = "eager"
    m = Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():                                  # sesgos y normas no triviales
        for n, p in m.named_parameters():
            if n.endswith("bias") or "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
    d = tmp / name
    m.save_pretrained(str(d))
    return d, m


@pytest.fixture
def pair(tmp_path):
    d_dir, d_model = _save(tmp_path, "donor", DONOR, 0)
    r_dir, _ = _save(tmp_path, "recip", RECIP, 1)
    return tmp_path, d_dir, d_model, r_dir


def _recip_cfg(r_dir):
    return json.loads((r_dir / "config.json").read_text())


def test_compat_real_pair_and_head_dim():
    assert check_compat(ArchSpec.from_config(QWEN_7B), ArchSpec.from_config(QWEN_15B)) == []
    half = dict(QWEN_15B, hidden_size=896, num_attention_heads=14, intermediate_size=4864, num_hidden_layers=24)
    errs = check_compat(ArchSpec.from_config(QWEN_15B), ArchSpec.from_config(half))   # 1.5B → 0.5B
    assert any("head_dim" in e for e in errs) and any("heads" in e for e in errs)


def test_im_is_front_aligned():
    s = select_im(ArchSpec.from_config(QWEN_7B), ArchSpec.from_config(QWEN_15B))
    assert s.layers == list(range(28)) and s.hidden == list(range(1536))
    assert s.q_heads[5] == list(range(12)) and s.kv_heads[5] == [0, 1] and s.mlp[0][-1] == 8959


def test_apm_selection_topk_and_gqa_rule():
    donor, recip = ArchSpec.from_config(dict(DONOR, model_type="qwen2")), ArchSpec.from_config(dict(RECIP, model_type="qwen2"))
    st = ActivationStats.zeros(donor)
    st.tokens = 1
    st.layer[:] = torch.tensor([1.0, 3.0, 2.0])                     # capas 1 y 2, en orden del donante
    st.hidden[:] = torch.arange(48.0).flip(0) * 3                   # los 32 primeros
    st.head[1] = torch.tensor([9, 0, 0, 1, 1, 1.])                  # Q: 0,3,4,5; KV: grupos {0,1,2}=9, {3,4,5}=3
    st.head[2] = torch.tensor([0, 0, 1, 5, 5, 5.])
    st.mlp[2] = torch.arange(96.0)
    s = select_apm(st, donor, recip)
    assert s.layers == [1, 2]
    assert s.hidden == list(range(32))
    assert s.q_heads[0] == [0, 3, 4, 5] and s.kv_heads[0] == [0, 1]
    assert s.q_heads[1] == [2, 3, 4, 5]
    assert s.mlp[1] == list(range(32, 96))
    assert s.digest() == select_apm(st, donor, recip).digest()      # determinista


def test_slice_shapes_load_strict_and_values(pair):
    tmp, d_dir, d_model, r_dir = pair
    donor = ArchSpec.from_path(d_dir)
    sel = random_selection(donor, ArchSpec.from_path(r_dir), seed=3)
    man = build_slice(d_dir, _recip_cfg(r_dir), sel, tmp / "slice")
    m = Qwen2ForCausalLM.from_pretrained(str(tmp / "slice")).eval()       # carga estricta con forma del receptor
    assert m.config.tie_word_embeddings and man["n_params"] > 0
    out = m(torch.tensor([[1, 2, 3, 4]])).logits
    assert out.shape == (1, 4, 120) and torch.isfinite(out).all()
    sd, dsd = m.state_dict(), d_model.state_dict()
    H = torch.tensor(sel.hidden)
    l0, qh = sel.layers[1], sel.q_heads[1][2]
    got = sd["model.layers.1.self_attn.q_proj.weight"][2 * 8:3 * 8]
    assert torch.equal(got, dsd[f"model.layers.{l0}.self_attn.q_proj.weight"][qh * 8:(qh + 1) * 8][:, H])
    kv = sel.kv_heads[0][1]
    assert torch.equal(sd["model.layers.0.self_attn.v_proj.bias"][8:16],
                       dsd[f"model.layers.{sel.layers[0]}.self_attn.v_proj.bias"][kv * 8:(kv + 1) * 8])
    n = sel.mlp[1][5]
    assert torch.equal(sd["model.layers.1.mlp.down_proj.weight"][:, 5],
                       dsd[f"model.layers.{l0}.mlp.down_proj.weight"][H, n])
    assert torch.equal(sd["model.embed_tokens.weight"], dsd["model.embed_tokens.weight"][:120][:, H])


def test_identity_selection_reproduces_donor(tmp_path):
    d_dir, d_model = _save(tmp_path, "d", RECIP, 5)
    sp = ArchSpec.from_path(d_dir)
    build_slice(d_dir, _recip_cfg(d_dir), select_im(sp, sp), tmp_path / "s")
    m = Qwen2ForCausalLM.from_pretrained(str(tmp_path / "s"))
    for k, v in d_model.state_dict().items():
        assert torch.equal(m.state_dict()[k], v), k


def test_merge_mu_endpoints_and_linearity(pair):
    tmp, d_dir, _, r_dir = pair
    sel = random_selection(ArchSpec.from_path(d_dir), ArchSpec.from_path(r_dir), seed=1)
    build_slice(d_dir, _recip_cfg(r_dir), sel, tmp / "slice")
    R = Qwen2ForCausalLM.from_pretrained(str(r_dir), dtype=torch.float32).state_dict()
    S = Qwen2ForCausalLM.from_pretrained(str(tmp / "slice"), dtype=torch.float32).state_dict()
    for mu in (0.0, 0.05, 1.0):
        m = Qwen2ForCausalLM.from_pretrained(str(r_dir), dtype=torch.float32)
        materialize(m, apm_recipe(str(r_dir), str(tmp / "slice"), mu))
        for k, v in m.state_dict().items():
            assert torch.allclose(v, (1 - mu) * R[k] + mu * S[k], atol=1e-6), (mu, k)


def test_originals_untouched_and_no_overwrite(pair):
    tmp, d_dir, _, r_dir = pair
    before = {p.name: sha256_file(p) for d in (d_dir, r_dir) for p in d.iterdir()}
    sel = select_im(ArchSpec.from_path(d_dir), ArchSpec.from_path(r_dir))
    build_slice(d_dir, _recip_cfg(r_dir), sel, tmp / "slice")
    with pytest.raises(FileExistsError):
        build_slice(d_dir, _recip_cfg(r_dir), sel, tmp / "slice")
    m = Qwen2ForCausalLM.from_pretrained(str(r_dir), dtype=torch.float32)
    materialize(m, apm_recipe(str(r_dir), str(tmp / "slice"), 0.1))
    assert before == {p.name: sha256_file(p) for d in (d_dir, r_dir) for p in d.iterdir()}


def test_streaming_profile_matches_full_model_hooks(pair):
    _, d_dir, d_model, _ = pair
    spec = ArchSpec.from_path(d_dir)
    g = torch.Generator().manual_seed(0)
    exs = []
    for T in (7, 11, 5):
        ids = torch.randint(0, 128, (T,), generator=g).tolist()
        exs.append((ids, [i >= 2 for i in range(T)]))                 # perfila solo una parte de los tokens
    got, cost = profile_donor(d_dir, exs, "cpu", dtype=torch.float32, log=lambda *_: None)
    ref = ActivationStats.zeros(spec)
    ref.tokens = sum(sum(m) for _, m in exs)
    cap = {}
    hooks = []
    for r, layer in enumerate(d_model.model.layers):
        hooks += [layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, r=r: cap.__setitem__(("a", r), a[0])),
                  layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, r=r: cap.__setitem__(("m", r), a[0])),
                  layer.register_forward_hook(lambda m, a, o, r=r: cap.__setitem__(("h", r), o[0] if isinstance(o, tuple) else o))]
    with torch.no_grad():
        for ids, mask in exs:
            d_model(torch.tensor([ids]))
            mk = torch.tensor(mask)
            for r in range(spec.layers):
                ref.update(r, cap[("h", r)][0][mk], cap[("a", r)][0][mk], cap[("m", r)][0][mk], spec.heads)
    for k in ("layer", "hidden", "head", "mlp"):
        assert torch.allclose(getattr(got, k), getattr(ref, k), rtol=1e-4, atol=1e-5), k
    assert got.tokens == ref.tokens == 17 and len(cost["layer_seconds"]) == 3
