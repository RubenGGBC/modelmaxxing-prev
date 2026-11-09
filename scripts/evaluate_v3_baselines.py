"""B, R, D: MBPP sanitized test[:100] (código) y MedMCQA validation[:300] 3-shot (medicina)."""
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_maxxing.evaluation.medical import eval_medmcqa, load_medmcqa
from model_maxxing.evaluation.runner import evaluate_mbpp

names = sys.argv[1:] or ["Qwen2.5-0.5B", "Med-Receiver", "Code-Donor"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
items = load_medmcqa("validation", 0, 300)
shots = load_medmcqa("train", 0, 3)
ex = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))[:100]
summary = {}
for n in names:
    p = f"models_cache/{n}"
    tok = AutoTokenizer.from_pretrained(p)
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float16).to(dev).eval()
    med = eval_medmcqa(m, tok, dev, items, shots, f"results/metrics/med_{n}.jsonl")
    del m
    summary[n] = {"med_acc": sum(r["correct"] for r in med) / len(med)}
    print(n, summary[n], flush=True)
    if n != "Qwen2.5-0.5B" or not Path("results/metrics/mbpp_Qwen2.5-0.5B.jsonl").exists():
        rows = evaluate_mbpp(p, ex, f"results/metrics/mbpp_{n}.jsonl")
        summary[n]["mbpp"] = sum(r["passed"] for r in rows) / len(rows)
    else:
        summary[n]["mbpp"] = sum(json.loads(l)["passed"] for l in open("results/metrics/mbpp_Qwen2.5-0.5B.jsonl")) / 100
    print(n, summary[n], flush=True)
json.dump(summary, open("results/metrics/v3_baselines.json", "w"), indent=1)
