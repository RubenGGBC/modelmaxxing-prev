"""Experimento decisivo v0.3: brazos con presupuesto idéntico (N0=1.0, <=12 down_proj, rango <=8).

  A   sin cirugía (R)                       B   SVD de Δ_D = D-B (12 capas con más energía de rango 8)
  B2  SVD de -Δ_R = -(R-B) (misma regla)    C   regla fija sobre la tabla de sensibilidad (uniforme)
  D   diseño de Claude (configs/arm_D_claude_design.json)
  E   control aleatorio igualado a D (mismas capas, rangos, escalas y norma total; seed 0)
Cada brazo X se evalúa solo (X) y con transferencia R* + 0.5·(D-B) (X_T). Referencia extra RB_05 = 0.5·R + 0.5·B.

Uso:  PYTHONPATH=src python scripts/run_arms.py [build|eval|report|all] [--only NOMBRE ...]
Todo en FP32 (los parches, ~1e-5 por elemento, se perderían en bf16). Reanudable: salta configuraciones con
results/arms/summary_<nombre>.json. Datos de evaluación separados de los visibles para el agente."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

from model_maxxing.surgery.arms import (make_spec, random_like, rescale, spec_from_matrix, total_norm, truncated,
                                        validate_budget)
from model_maxxing.surgery.patch_builder import patch_tensors
from model_maxxing.surgery.patch_store import load_patch, save_patch
from model_maxxing.surgery.patch_validator import validate_patch

N0, M, K, ALPHA_T = 1.0, 12, 8, 0.5
OUT = Path("results/arms")
PATCH_DIR = str(OUT / "patches")
MODELS = {"B": "models_cache/Qwen2.5-0.5B", "R": "models_cache/Med-Receiver", "D": "models_cache/Code-Donor"}
D_CONFIG = "configs/arm_D_claude_design.json"
SENS = "results/metrics/v3_sensitivity_table.json"
ORDER = ["A", "A_T", "D", "D_T", "E", "E_T", "C", "C_T", "B", "B_T", "B2", "B2_T", "RB_05"]


def T(i: int) -> str:
    return f"model.layers.{i}.mlp.down_proj.weight"


def load_down(path: str) -> dict:
    with safe_open(f"{path}/model.safetensors", "pt") as f:
        return {T(i): f.get_tensor(T(i)).float() for i in range(24)}


def unit(M_: torch.Tensor) -> torch.Tensor:
    P = truncated(M_, K)
    return P / P.norm()


def edits_with_norm(t: str, M_: torch.Tensor, norm: float, author: str) -> list:
    return spec_from_matrix(t, unit(M_) * norm, K, author)


# ---------------------------------------------------------------- construcción de brazos

def svd_arm(delta: dict, author: str, why: str):
    """Regla convencional: 12 tensores con mayor norma de su truncado rango 8; escala común a norma total N0."""
    tn = {t: truncated(d, K).norm().item() for t, d in delta.items()}
    top = sorted(tn, key=lambda t: -tn[t])[:M]
    f = N0 / math.sqrt(sum(tn[t] ** 2 for t in top))
    edits = [e for t in top for e in edits_with_norm(t, delta[t], tn[t] * f, author)]
    return make_spec(edits, author, why, meta={"tensors": top, "norms": {t: tn[t] * f for t in top}})


def c_arm(dirs: dict, table: dict):
    """Regla fija sin Claude: por capa, dirección con menor d_code+d_bio; 12 mejores; norma uniforme N0/sqrt(12)."""
    best = {int(l): min((r[k]["d_code"] + r[k]["d_bio"], k) for k in ("dR-", "dD+")) for l, r in table.items()}
    top = sorted(best, key=lambda l: best[l][0])[:M]
    n = N0 / math.sqrt(M)
    edits = [e for l in top for e in edits_with_norm(T(l), dirs[best[l][1]][T(l)], n, "rule_C")]
    return make_spec(edits, "rule_C", "regla fija: mejor (d_code+d_bio) por capa, top-12, norma uniforme",
                     meta={"layers": sorted(top), "directions": {l: best[l][1] for l in top}})


def d_arm(dirs: dict, table: dict, cfg: dict):
    rule = cfg["rule"]
    lam, cap = rule["lambda"], rule["cap_per_tensor"]
    g = {int(l): {k: max(0.0, -(r[k]["d_code"] + lam * r[k]["d_bio"])) for k in ("dR-", "dD+")} for l, r in table.items()}
    G = {l: math.sqrt(sum(v * v for v in gl.values())) for l, gl in g.items()}
    top = sorted(G, key=lambda l: -G[l])[:M]
    if sorted(top) != sorted(cfg["expected_layers"]):
        raise SystemExit(f"Capas de D {sorted(top)} != pre-registradas {cfg['expected_layers']}")
    # reparto proporcional a G con tope por tensor (iterativo)
    fixed, free = {}, list(top)
    while True:
        budget = math.sqrt(max(N0 ** 2 - sum(v * v for v in fixed.values()), 0.0))
        s = math.sqrt(sum(G[l] ** 2 for l in free))
        alloc = {l: budget * G[l] / s for l in free}
        over = [l for l in free if alloc[l] > cap]
        if not over:
            break
        for l in over:
            fixed[l] = cap
            free.remove(l)
    norms = {**fixed, **alloc}
    edits = []
    for l in top:
        Ml = sum(g[l][k] * unit(dirs[k][T(l)]) for k in ("dR-", "dD+"))
        edits += edits_with_norm(T(l), Ml, norms[l], "claude_D")
    return make_spec(edits, "claude_D", "diseño de Claude: ver " + D_CONFIG,
                     meta={"layers": sorted(top), "norms": {l: norms[l] for l in sorted(top)},
                           "gains": {l: g[l] for l in sorted(top)}})


def build() -> dict:
    Bd, Rd, Dd = (load_down(MODELS[n]) for n in "BRD")
    dD = {t: Dd[t] - Bd[t] for t in Bd}
    mR = {t: -(Rd[t] - Bd[t]) for t in Bd}
    table = json.load(open(SENS))["table"]
    cfg = json.load(open(D_CONFIG))
    dirs = {"dD+": dD, "dR-": mR}
    specs = {
        "B": svd_arm(dD, "svd_dD", "SVD convencional de D-B"),
        "B2": svd_arm(mR, "svd_mR", "SVD convencional de -(R-B)"),
        "C": c_arm(dirs, table),
        "D": d_arm(dirs, table, cfg),
    }
    e = random_like(specs["D"], seed=0)
    specs["E"] = rescale_spec(e, total_norm(list(specs["D"].edits)) / total_norm(list(e.edits)))
    man = {}
    for name, sp in specs.items():
        errs = validate_budget(sp, M, K, N0)
        v = validate_patch(sp, Rd)
        if errs or not v.ok:
            raise SystemExit(f"Brazo {name} inválido: {errs} {v.errors}")
        pid = save_patch(sp, PATCH_DIR)
        man[name] = {"patch_id": pid, "author": sp.author, "total_norm": round(v.total_norm, 6),
                     "tensors": sorted({e.tensor for e in sp.edits}, key=lambda t: int(t.split(".")[2])),
                     "per_tensor_norm": {t: round(n * Rd[t].norm().item(), 6) for t, n in v.relative_norms.items()},
                     "meta": sp.meta}
        print(name, pid, "norma", round(v.total_norm, 4), "capas", [int(t.split(".")[2]) for t in man[name]["tensors"]])
    # solapamiento de direcciones entre brazos (coseno de los deltas densos)
    dense = {n: patch_tensors(sp, Rd) for n, sp in specs.items()}
    def cos(a, b):
        num = sum((a[t] * b[t]).sum().item() for t in a if t in b)
        return num / (math.sqrt(sum(x.pow(2).sum().item() for x in a.values())) *
                      math.sqrt(sum(x.pow(2).sum().item() for x in b.values())))
    names = list(specs)
    man["_cosines"] = {f"{a}~{b}": round(cos(dense[a], dense[b]), 4) for i, a in enumerate(names) for b in names[i + 1:]}
    print("cosenos", man["_cosines"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def rescale_spec(sp, factor: float):
    return make_spec(rescale(list(sp.edits), factor), sp.author, sp.justification, meta=sp.meta)


# ---------------------------------------------------------------- evaluación

def config_of(name: str) -> tuple[str | None, float, bool]:
    """(brazo cuyo parche se aplica, alpha de transferencia, ¿es R->B?)"""
    if name == "RB_05":
        return None, 0.0, True
    arm, _, tr = name.partition("_")
    return (None if arm == "A" else arm), (ALPHA_T if tr == "T" else 0.0), False


def evaluate(only: list[str] | None) -> None:
    from datasets import load_dataset
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from model_maxxing.agents.diagnostics import corpus_nll
    from model_maxxing.evaluation.medical import eval_medmcqa, load_medmcqa
    from model_maxxing.evaluation.runner import best_device, evaluate_mbpp_model, run_tests

    ok, err = run_tests("x = 1", {"test_list": ["assert x == 1"]})
    bad, _ = run_tests("x = 1", {"test_list": ["assert x == 2"]})
    if not ok or bad:
        raise SystemExit(f"Sandbox no operativo: ok={ok} bad={bad} {err}")
    man = json.loads((OUT / "manifest.json").read_text())
    dev = best_device()
    torch.manual_seed(0)
    # datos de EVALUACIÓN (separados de los visibles: MBPP full train, PubMedQA[100:160], MedMCQA train)
    mbpp = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))[:100]
    items, shots = load_medmcqa("validation", 0, 300), load_medmcqa("train", 0, 3)
    pq = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    bio = ["\n".join(pq[i]["context"]["contexts"]) for i in range(200, 260)]
    code = [ex["code"] for ex in mbpp]

    W = {n: load_file(f"{MODELS[n]}/model.safetensors", device=dev) for n in "BRD"}
    tok = AutoTokenizer.from_pretrained(MODELS["R"])
    model = AutoModelForCausalLM.from_pretrained(MODELS["R"], dtype=torch.float32).to(dev).eval()
    params = dict(model.named_parameters())
    missing = set(params) - set(W["R"])
    if missing:
        raise SystemExit(f"Parámetros sin peso en el checkpoint: {sorted(missing)[:5]}")

    for name in ORDER:
        if only and name not in only:
            continue
        out = OUT / f"summary_{name}.json"
        if out.exists():
            print("ya hecho:", name)
            continue
        arm, alpha, rb = config_of(name)
        deltas = patch_tensors(load_patch(man[arm]["patch_id"], PATCH_DIR), W["R"]) if arm else {}
        applied = 0.0
        with torch.no_grad():
            for k, p in params.items():
                r = W["R"][k].float()
                w = 0.5 * r + 0.5 * W["B"][k].float() if rb else r + alpha * (W["D"][k].float() - W["B"][k].float()) if alpha else r.clone()
                ref = w.clone() if k in deltas else None
                if k in deltas:
                    w += deltas[k].to(dev)
                p.data.copy_(w)
                if ref is not None:
                    applied += (p.data - ref).pow(2).sum().item()
        t0 = time.time()
        res = {"name": name, "arm": arm or "none", "alpha_transfer": alpha, "R_to_B": rb,
               "patch_id": man[arm]["patch_id"] if arm else None, "applied_patch_norm": round(math.sqrt(applied), 6),
               "device": dev, "dtype": "float32"}
        res["nll_bio"] = round(corpus_nll(model, tok, dev, bio), 5)
        res["nll_code"] = round(corpus_nll(model, tok, dev, code), 5)
        med = eval_medmcqa(model, tok, dev, items, shots, str(OUT / f"med_{name}.jsonl"))
        res["med_acc"] = sum(r["correct"] for r in med) / len(med)
        print(name, {k: res[k] for k in ("applied_patch_norm", "nll_bio", "nll_code", "med_acc")}, flush=True)
        rows = evaluate_mbpp_model(model, tok, dev, mbpp, str(OUT / f"mbpp_{name}.jsonl"), tag=name)
        res["mbpp_pass"] = sum(r["passed"] for r in rows)
        res["mbpp_n"] = len(rows)
        res["seconds"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(res, indent=1))
        print(name, "MBPP", res["mbpp_pass"], "/", len(rows), f"({res['seconds']} s)", flush=True)


# ---------------------------------------------------------------- informe

def report() -> dict:
    from model_maxxing.evaluation.metrics import paired_bootstrap

    S = {n: json.loads((OUT / f"summary_{n}.json").read_text()) for n in ORDER if (OUT / f"summary_{n}.json").exists()}
    per = lambda kind, n, key: [json.loads(l)[key] for l in open(OUT / f"{kind}_{n}.jsonl") if l.strip()]
    A = S["A"]
    rows = {}
    for n, s in S.items():
        valid = s["med_acc"] >= A["med_acc"] - 0.05 and s["nll_bio"] <= A["nll_bio"] + 0.02
        rows[n] = {k: s[k] for k in ("mbpp_pass", "med_acc", "nll_bio", "nll_code", "applied_patch_norm")}
        rows[n]["valid"] = valid
    pairs = [(n, "A") for n in S if n != "A"] + [(n, "A_T") for n in S if n.endswith("_T") and n != "A_T"]
    pairs += [(f"D{s}", f"{o}{s}") for s in ("", "_T") for o in ("E", "C", "B", "B2")]
    cmp = {}
    for x, ref in pairs:
        if x in S and ref in S:
            cmp[f"{x} vs {ref}"] = {
                "mbpp": paired_bootstrap(per("mbpp", ref, "passed"), per("mbpp", x, "passed")),
                "med": paired_bootstrap(per("med", ref, "correct"), per("med", x, "correct")),
                "d_nll_bio": round(S[x]["nll_bio"] - S[ref]["nll_bio"], 5),
                "d_nll_code": round(S[x]["nll_code"] - S[ref]["nll_code"], 5)}
    cands = [n for n in S if n not in ("A", "RB_05") and rows[n]["valid"]]
    best = max(cands, key=lambda n: rows[n]["mbpp_pass"]) if cands else None
    rep = {"criterion": "válido si MedMCQA >= A - 5 pp y NLL bio <= A + 0.02; mejor MBPP solo entre válidos",
           "rows": rows, "comparisons": cmp, "best_valid": best}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"{'config':8} {'MBPP':>5} {'MedMCQA':>8} {'NLLbio':>8} {'NLLcode':>8} {'|P|':>6} válido")
    for n, r in rows.items():
        print(f"{n:8} {r['mbpp_pass']:>5} {r['med_acc']:>8.3f} {r['nll_bio']:>8.4f} {r['nll_code']:>8.4f} "
              f"{r['applied_patch_norm']:>6.3f} {r['valid']}")
    for k, c in cmp.items():
        m = c["mbpp"]
        print(f"{k:14} MBPP {m['diff']*100:+.0f} pp [{m['ci_low']*100:+.0f}, {m['ci_high']*100:+.0f}]  "
              f"Med {c['med']['diff']*100:+.1f} pp  ΔNLLbio {c['d_nll_bio']:+.4f}  ΔNLLcode {c['d_nll_code']:+.4f}")
    print("mejor válido:", best)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="all", choices=["build", "eval", "report", "all"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.step in ("build", "all") and not (a.step == "all" and (OUT / "manifest.json").exists()):
        build()
    if a.step in ("eval", "all"):
        evaluate(a.only)
    if a.step in ("report", "all"):
        report()
