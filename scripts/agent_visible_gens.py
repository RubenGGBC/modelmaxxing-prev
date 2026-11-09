"""Generaciones de R y D en el split visible para el agente: MBPP full train[:60]. NO se usa en evaluación final."""
import sys
from datasets import load_dataset
from model_maxxing.evaluation.runner import evaluate_mbpp
ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
ex = [dict(ds[i], prompt=ds[i]["text"], test_imports=ds[i].get("test_setup_code") and [ds[i]["test_setup_code"]] or []) for i in range(60)]
for n in sys.argv[1:]:
    rows = evaluate_mbpp(f"models_cache/{n}", ex, f"results/metrics/visible_mbpp_{n}.jsonl")
    print(n, sum(r["passed"] for r in rows), "/", len(rows), flush=True)
