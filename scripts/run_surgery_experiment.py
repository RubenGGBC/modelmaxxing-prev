"""Variantes v0.2 (receptor sustituto = base 0.5B). Uso: python scripts/run_surgery_experiment.py N PATCH_ID ALPHA
A: Transfer(R,D)  C: Transfer(R+P,D)  D: Transfer(R+Pctrl,D)  E: R+P   (+ Dctrl: R+Pctrl sin transferencia)
Control: mismo tensor/bloque/rango/norma, direcciones u,v aleatorias (seed fija). ALPHA fijado antes de ejecutar."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file

from model_maxxing.evaluation.runner import evaluate_mbpp
from model_maxxing.surgery.patch_builder import apply_patch
from model_maxxing.surgery.patch_schema import LowRankEdit, PatchSpec
from model_maxxing.surgery.patch_store import load_patch, save_patch
from model_maxxing.transfer.engine import survival, transfer

n, pid, alpha = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
RD, DD = "models_cache/Qwen2.5-0.5B", "models_cache/Qwen2.5-Coder-0.5B"
R, D = load_file(f"{RD}/model.safetensors"), load_file(f"{DD}/model.safetensors")
P = load_patch(pid)
e = P.edits[0]
g = torch.Generator().manual_seed(42)
u, v = torch.randn(len(e.u), generator=g), torch.randn(len(e.v), generator=g)
u, v = u / u.norm(), v / v.norm()
Pc = PatchSpec(edits=(LowRankEdit(e.tensor, e.row0, e.col0, tuple(u.tolist()), tuple(v.tolist()), e.scale),),
               author="control_random", parent_id=P.parent_id, justification="control igualado en norma")
cid = save_patch(Pc)
Rp, Rc = apply_patch(R, P), apply_patch(R, Pc)
variants = {
    "A_transfer": transfer(R, D, "interpolate", alpha),
    "C_transfer_Pclaude": transfer(Rp, D, "interpolate", alpha),
    "D_transfer_Pcontrol": transfer(Rc, D, "interpolate", alpha),
    "E_Pclaude_only": Rp,
    "Dctrl_Pcontrol_only": Rc,
}
T = e.tensor
surv = {"C_vs_A": survival(variants["C_transfer_Pclaude"], variants["A_transfer"], [T])[T],
        "D_vs_A": survival(variants["D_transfer_Pcontrol"], variants["A_transfer"], [T])[T],
        "patch_norm_direct": (Rp[T].float() - R[T].float()).norm().item(),
        "R_star_differs": bool(not torch.equal(Rp[T], R[T]))}
print("survival", surv, flush=True)
ex = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))[:n]
res = {"patch": pid, "control_patch": cid, "alpha": alpha, "n": n, "survival": surv, "passed": {}}
for name, st in variants.items():
    with tempfile.TemporaryDirectory(dir="models_cache") as tmp:
        for f in list(Path(RD).glob("*.json")) + list(Path(RD).glob("merges.txt")):
            shutil.copy(f, tmp)
        save_file(st, f"{tmp}/model.safetensors", metadata={"format": "pt"})
        rows = evaluate_mbpp(tmp, ex, f"results/metrics/mbpp_surg_{name}.jsonl")
    res["passed"][name] = sum(r["passed"] for r in rows)
    print(name, res["passed"][name], "/", len(rows), flush=True)
    Path("results/metrics/surgery_experiment.json").write_text(json.dumps(res, indent=1))
