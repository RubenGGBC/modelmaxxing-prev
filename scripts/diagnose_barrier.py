"""Prioridad 1 (v0.3): diagnóstico de la barrera de interpolación R(base 0.5B) <-> D(Coder 0.5B).
Uso: python scripts/diagnose_barrier.py   -> results/reports/barrier_report.json
Todo en memoria: un solo modelo HF al que se le cargan los pesos interpolados."""
import json
import re

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_maxxing.transfer.interpolation import interpolate

RD, DD = "models_cache/Qwen2.5-0.5B", "models_cache/Qwen2.5-Coder-0.5B"
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
rep = {"device": dev}

R, D = load_file(f"{RD}/model.safetensors"), load_file(f"{DD}/model.safetensors")
rep["dtypes"] = {"R": sorted({str(v.dtype) for v in R.values()}), "D": sorted({str(v.dtype) for v in D.values()})}

# ---- 1. Pesos compartidos / embeddings
cfg = {k: json.load(open(f"{p}/config.json")).get(k) for p in (RD, DD) for k in ("tie_word_embeddings",)}
rep["tie_word_embeddings"] = {"R": json.load(open(f"{RD}/config.json"))["tie_word_embeddings"],
                              "D": json.load(open(f"{DD}/config.json"))["tie_word_embeddings"]}
rep["lm_head_in_checkpoint"] = {"R": "lm_head.weight" in R, "D": "lm_head.weight" in D}
rep["tensor_count"] = {"R": len(R), "D": len(D)}
e = "model.embed_tokens.weight"
rep["embedding"] = {"shape": list(R[e].shape), "R_vs_D_rel_diff": ((D[e].float() - R[e].float()).norm() / R[e].float().norm()).item()}

# ---- 2. Verificación matemática de los pesos interpolados
math = {}
for a in ALPHAS:
    W = interpolate(R, D, a)
    max_err_formula, max_err_fp16round, worst = 0.0, 0.0, ""
    for k, r in R.items():
        if not r.is_floating_point():
            continue
        w32 = (1 - a) * r.float() + a * D[k].float()
        alt = r.float() + a * (D[k].float() - r.float())          # formulación independiente
        max_err_formula = max(max_err_formula, (w32 - alt).abs().max().item())
        err = (W[k].float() - w32).abs().max().item()
        if err > max_err_fp16round:
            max_err_fp16round, worst = err, k
    endpoints = None
    if a == 0.0:
        endpoints = all(torch.equal(W[k], R[k]) for k in R)
    if a == 1.0:
        endpoints = all(torch.equal(W[k], D[k]) for k in D)
    math[str(a)] = {"max_abs_err_two_formulas_fp32": max_err_formula,
                    "max_abs_err_fp16_rounding": max_err_fp16round, "worst_tensor": worst,
                    "endpoint_bitexact": endpoints,
                    "finite": all(torch.isfinite(v).all().item() for v in W.values() if v.is_floating_point())}
rep["math_check"] = math

# ---- 3. Corpus fijo para NLL
tok = AutoTokenizer.from_pretrained(RD)
code = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
code_text = "\n\n".join(code[i]["code"] for i in range(100, 160))          # fuera de los 100 de evaluación
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
wiki_text = "\n".join(t for t in wt["text"][:400] if t.strip())
def chunks(text, n=24, L=512):
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * L:(i + 1) * L] for i in range(min(n, len(ids) // L))]
CORP = {"code": chunks(code_text), "text": chunks(wiki_text)}
rep["corpus"] = {k: {"chunks": len(v), "tokens": sum(len(c) for c in v)} for k, v in CORP.items()}

model = AutoModelForCausalLM.from_pretrained(RD, torch_dtype=torch.float32).to(dev).eval()

@torch.no_grad()
def nll(state, dtype):
    missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=False)
    model.to(dtype)
    out = {"missing": list(missing), "unexpected": list(unexpected)}
    for name, cs in CORP.items():
        tot, n = 0.0, 0
        for c in cs:
            ids = c.unsqueeze(0).to(dev)
            logits = model(ids).logits.float()
            tot += torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item()
            n += ids.shape[1] - 1
        out[name] = tot / n
    return out

# ---- 4. NLL vs alpha, FP32 vs FP16
curve = {}
for a in ALPHAS:
    W32 = {k: ((1 - a) * R[k].float() + a * D[k].float()) if R[k].is_floating_point() else R[k] for k in R}
    W16 = interpolate(R, D, a)
    r32, r16 = nll(W32, torch.float32), nll(W16, torch.float16)
    curve[str(a)] = {"fp32": {k: r32[k] for k in CORP}, "fp16": {k: r16[k] for k in CORP},
                     "missing_keys": r32["missing"], "unexpected_keys": r32["unexpected"]}
    print(a, curve[str(a)]["fp32"], curve[str(a)]["fp16"], flush=True)
rep["nll_vs_alpha"] = curve

# barrera = NLL(alpha) - interpolación lineal de los extremos
bar = {}
for name in CORP:
    n0, n1 = curve["0.0"]["fp32"][name], curve["1.0"]["fp32"][name]
    bar[name] = {str(a): curve[str(a)]["fp32"][name] - ((1 - a) * n0 + a * n1) for a in ALPHAS}
rep["loss_barrier_vs_linear_endpoints_fp32"] = bar

# ---- 5. Localización: alpha=0.5 aplicado solo a un grupo de tensores
groups = {"embed": r"embed_tokens", "attn": r"self_attn", "mlp": r"\.mlp\.", "norms": r"layernorm|model\.norm"}
loc = {}
for g, pat in groups.items():
    W = {k: (0.5 * R[k].float() + 0.5 * D[k].float()) if (re.search(pat, k) and R[k].is_floating_point()) else R[k].float()
         for k in R}
    r = nll(W, torch.float32)
    loc[g] = {k: r[k] for k in CORP}
    print("solo", g, loc[g], flush=True)
rep["alpha0.5_only_group_fp32"] = loc
for half, rng in {"layers_0_11": range(0, 12), "layers_12_23": range(12, 24)}.items():
    W = {k: (0.5 * R[k].float() + 0.5 * D[k].float())
         if (R[k].is_floating_point() and (m := re.search(r"layers\.(\d+)\.", k)) and int(m.group(1)) in rng) else R[k].float()
         for k in R}
    r = nll(W, torch.float32)
    loc[half] = {k: r[k] for k in CORP}
    print("solo", half, loc[half], flush=True)

json.dump(rep, open("results/reports/barrier_report.json", "w"), indent=1)
print("OK")
