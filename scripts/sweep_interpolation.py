"""Uso: python scripts/sweep_interpolation.py N RECEIVER_DIR DONOR_DIR ALPHA [ALPHA...]
Interpola, guarda el checkpoint en un tmp, evalúa MBPP[:N] y lo borra. Guarda la receta, no el checkpoint."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from datasets import load_dataset
from safetensors.torch import load_file, save_file

from model_maxxing.evaluation.runner import evaluate_mbpp
from model_maxxing.transfer.interpolation import interpolate

n, recv, donor, *alphas = sys.argv[1:]
n = int(n)
ex = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))[:n]
R = load_file(str(Path(recv) / "model.safetensors"))
D = load_file(str(Path(donor) / "model.safetensors"))
Path("results/metrics").mkdir(parents=True, exist_ok=True)
for a in map(float, alphas):
    with tempfile.TemporaryDirectory(dir="/Users/rubengallego/ModelMaxxing/models_cache") as tmp:
        for f in Path(recv).glob("*.json"):
            shutil.copy(f, tmp)
        for f in Path(recv).glob("merges.txt"):
            shutil.copy(f, tmp)
        save_file(interpolate(R, D, a), f"{tmp}/model.safetensors", metadata={"format": "pt"})
        tag = f"interp_a{a:.2f}"
        rows = evaluate_mbpp(tmp, ex, f"results/metrics/mbpp_{tag}.jsonl")
    p = sum(r["passed"] for r in rows)
    Path(f"results/metrics/recipe_{tag}.json").write_text(json.dumps(
        {"operator": "interpolate", "alpha": a, "receiver": recv, "donor": donor, "n": n, "passed": p}))
    print(tag, p, "/", len(rows), flush=True)
