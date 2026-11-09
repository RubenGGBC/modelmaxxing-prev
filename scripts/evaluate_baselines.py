"""Uso: python scripts/evaluate_baselines.py N MODEL_DIR [MODEL_DIR...]  (MBPP sanitized, test[:N], determinista)"""
import sys
from pathlib import Path

from datasets import load_dataset

from model_maxxing.evaluation.runner import evaluate_mbpp

n = int(sys.argv[1])
ex = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))[:n]
Path("results/metrics").mkdir(parents=True, exist_ok=True)
for m in sys.argv[2:]:
    rows = evaluate_mbpp(m, ex, f"results/metrics/mbpp_{Path(m).name}.jsonl")
    print(m, sum(r["passed"] for r in rows), "/", len(rows))
