"""Tabla de sensibilidad (datos visibles para el agente): efecto en NLL(bio, code) de parches de rango 8
sobre down_proj de cada capa, norma fija por tensor n_t = N0/sqrt(12), para dos direcciones:
  dD+ : truncado SVD de (D-B) ;  dR- : truncado SVD de -(R-B)."""
import json
import math

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_maxxing.agents.diagnostics import patch_sensitivity
from model_maxxing.surgery.arms import truncated

N0, M, K = 1.0, 12, 8
dev = "mps"
pq = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
bio = ["\n".join(pq[i]["context"]["contexts"]) for i in range(100, 140)]
mb = load_dataset("google-research-datasets/mbpp", "full", split="train")
code = [mb[i]["code"] for i in range(0, 40)]
B, R, D = [load_file(f"models_cache/{n}/model.safetensors") for n in ["Qwen2.5-0.5B", "Med-Receiver", "Code-Donor"]]
tok = AutoTokenizer.from_pretrained("models_cache/Med-Receiver")
model = AutoModelForCausalLM.from_pretrained("models_cache/Med-Receiver", dtype=torch.float32).to(dev).eval()
nt = N0 / math.sqrt(M)
table = {}
for i in range(24):
    t = f"model.layers.{i}.mlp.down_proj.weight"
    row = {}
    for name, d in {"dD+": D[t].float() - B[t].float(), "dR-": -(R[t].float() - B[t].float())}.items():
        P = truncated(d, K)
        P = P * (nt / P.norm())
        r = patch_sensitivity(model, tok, dev, t, P, [0.0, 1.0], {"bio": bio, "code": code})
        row[name] = {"d_bio": round(r["1.0"]["bio"] - r["0.0"]["bio"], 5), "d_code": round(r["1.0"]["code"] - r["0.0"]["code"], 5)}
    row["base_nll"] = r["0.0"]
    table[i] = row
    print(i, row, flush=True)
json.dump({"N0": N0, "M": M, "K": K, "per_tensor_norm": nt, "table": table}, open("results/metrics/v3_sensitivity_table.json", "w"), indent=1)
