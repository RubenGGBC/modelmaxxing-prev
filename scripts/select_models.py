"""Hito A (v0.4): selección del par 1.5B en la partición de SELECCIÓN (nunca en test final).

Configuraciones (todas en bf16, sin módulos en FP32: son referencias naturales de cada modelo):
  B       Qwen2.5-1.5B                       R      Qwen2.5-1.5B-Instruct (receptor)
  D       Qwen2.5-Coder-1.5B (donante)       T_05   R + 0.5·(D−B)      T_10   R + 1.0·(D−B)
  RB_05   0.5·R + 0.5·B (control de vuelta a la base)
Fases: `nll` (NLL de la solución de referencia dada la consigna, por problema), `pass` (pass@1 greedy,
256 tokens, sandbox Docker), `weights` (normas y cosenos de Δ_R, Δ_D), `report`. Reanudable por problema.

Uso:  PYTHONPATH=src python scripts/select_models.py [weights|nll|pass|report|all] [--only NOMBRE ...]
        [--out DIR] [--batch-size N] [--dtype bf16|fp16]
--batch-size 1 = generación de uno en uno (modo de la ejecución local); >1 = por lotes. No mezclar modos en una
misma carpeta de salida: cada entorno/modo escribe en su propio --out."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from model_maxxing.evaluation.runner import (STOP, build_prompt, evaluate_mbpp_batched, evaluate_mbpp_model, run_tests,
                                            sandbox_description)
from model_maxxing.evaluation.splits import check_disjoint, ids_hash, load_split
from model_maxxing.models.loading import (CheckpointReader, Recipe, dtype_report, load_for_variants, materialize,
                                          memory_report)

OUT = Path("results/experiments/maxxing_v04/selection")
M = {"B": "models_cache/Qwen2.5-1.5B", "R": "models_cache/Qwen2.5-1.5B-Instruct", "D": "models_cache/Qwen2.5-Coder-1.5B"}
CONFIGS = {
    "B": Recipe(M["B"]), "R": Recipe(M["R"]), "D": Recipe(M["D"]),
    "T_05": Recipe(M["R"], "task_vector", 0.5, M["D"], M["B"]),
    "T_10": Recipe(M["R"], "task_vector", 1.0, M["D"], M["B"]),
    "RB_05": Recipe(M["R"], "interpolate", 0.5, M["B"]),
}


def selection_data() -> list[dict]:
    parts = {k: load_split(k) for k in ("diagnostic", "selection", "final_mbpp")}
    check_disjoint(parts)
    return parts["selection"]


def _done(path: Path) -> dict:
    if not path.exists():
        return {}
    return {json.loads(l)["task_id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


@torch.no_grad()
def solution_nll(model, tok, dev, ex: dict) -> tuple[float, int]:
    """Suma de NLL y nº de tokens de `code + ```` dada la consigna (mismo prompt que en generación)."""
    prompt = build_prompt(ex)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + ex["code"].rstrip() + "\n" + STOP, return_tensors="pt").input_ids
    n_p = p_ids.shape[1]
    if not torch.equal(full[0, :n_p], p_ids[0]):     # la frontera de tokenización debe coincidir
        n_p -= 1
    ids = full[:, :1024].to(dev)
    lg = model(ids).logits.float()
    tgt = ids[0, n_p:]
    nll = torch.nn.functional.cross_entropy(lg[0, n_p - 1:-1], tgt, reduction="sum").item()
    return nll, int(tgt.numel())


def weights() -> dict:
    """Normas por tensor de Δ_R = R−B y Δ_D = D−B, coseno global y por tipo de módulo (CPU, streaming)."""
    out = {"per_kind": {}, "global": {}}
    acc = {"nR": 0.0, "nD": 0.0, "dot": 0.0, "nB": 0.0}
    with CheckpointReader(M["B"]) as B, CheckpointReader(M["R"]) as R, CheckpointReader(M["D"]) as D:
        for k in B.keys():
            b = B.get(k).float()
            dr, dd = R.get(k).float() - b, D.get(k).float() - b
            kind = k.split(".")[-2] if "layers" in k else k.rsplit(".", 1)[0]
            s = out["per_kind"].setdefault(kind, {"nR": 0.0, "nD": 0.0, "dot": 0.0, "nB": 0.0})
            for a_, v in (("nR", dr.pow(2).sum()), ("nD", dd.pow(2).sum()), ("dot", (dr * dd).sum()), ("nB", b.pow(2).sum())):
                s[a_] += v.item()
                acc[a_] += v.item()
    fmt = lambda s: {"norm_dR": math.sqrt(s["nR"]), "norm_dD": math.sqrt(s["nD"]), "norm_B": math.sqrt(s["nB"]),
                     "rel_dR": math.sqrt(s["nR"] / s["nB"]), "rel_dD": math.sqrt(s["nD"] / s["nB"]),
                     "cos_dR_dD": s["dot"] / math.sqrt(s["nR"] * s["nD"] + 1e-30)}
    res = {"global": fmt(acc), "per_kind": {k: fmt(v) for k, v in out["per_kind"].items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["global"], indent=1))
    return res


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def run(phase: str, only: list[str] | None, batch_size: int = 1, dtype: str = "bf16") -> None:
    data = selection_data()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_manifest.json").write_text(json.dumps(
        {"split": "selection", "n": len(data), "ids_sha256_16": ids_hash(data),
         "sources": sorted({e["source"] for e in data})}, indent=1))
    if phase == "pass":
        ok, err = run_tests("x = 1", {"test_list": ["assert x == 1"]})
        bad, _ = run_tests("x = 1", {"test_list": ["assert x == 2"]})
        if not ok or bad:
            raise SystemExit(f"Sandbox no operativo: ok={ok} bad={bad} {err}")
    dev = "cuda"
    model, tok = load_for_variants(M["R"], dev, fp32_modules=[], dtype=DTYPES[dtype])
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": dtype,
           "gen_mode": f"batch{batch_size}" if batch_size > 1 else "single", "sandbox": sandbox_description()}
    envf = OUT / "environment.json"
    if envf.exists() and json.loads(envf.read_text()) != env:
        raise SystemExit(f"Entorno distinto al de {envf}: {json.loads(envf.read_text())} != {env}. Usa otro --out.")
    envf.write_text(json.dumps(env, indent=1))
    for name, rec in CONFIGS.items():
        if only and name not in only:
            continue
        path = OUT / f"{phase}_{name}.jsonl"
        done = _done(path)
        todo = [e for e in data if e["task_id"] not in done]
        if not todo:
            print("ya hecho:", phase, name, flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        info = materialize(model, rec)
        t0 = time.time()
        if phase == "nll":
            with open(path, "a", encoding="utf-8") as fh:
                for ex in todo:
                    nll, n = solution_nll(model, tok, dev, ex)
                    if nll != nll or nll == float("inf"):
                        raise SystemExit(f"NLL no finita en {name}/{ex['task_id']} (¿desbordamiento en {dtype}?)")
                    fh.write(json.dumps({"task_id": ex["task_id"], "nll_sum": nll, "n_tokens": n}) + "\n")
        elif batch_size > 1:
            def save(rows, path=path):
                with open(path, "a", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row) + "
")
            evaluate_mbpp_batched(model, tok, dev, todo, tag=name, batch_size=batch_size, on_rows=save)
        else:
            tmp = OUT / f"_tmp_{name}.jsonl"
            for ex in todo:     # de uno en uno para poder reanudar
                row = evaluate_mbpp_model(model, tok, dev, [ex], str(tmp), tag=name)[0]
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            tmp.unlink(missing_ok=True)
        meta = {"name": name, "recipe": rec.describe(), **info, "seconds": round(time.time() - t0, 1),
                "memory": memory_report(), "dtypes": dtype_report(model)}
        (OUT / f"meta_{phase}_{name}.json").write_text(json.dumps(meta, indent=1))
        print(phase, name, f"{meta['seconds']} s", meta["memory"].get("peak_allocated_gb"), "GB pico", flush=True)


def report() -> dict:
    from model_maxxing.diagnostics.capability_gap import gap_by_subskill
    from model_maxxing.evaluation.metrics import paired_bootstrap

    data = selection_data()
    ids = [e["task_id"] for e in data]
    res = {"n": len(ids), "configs": {}, "comparisons": {}}
    P, N = {}, {}
    for name in CONFIGS:
        r = {}
        nl = _done(OUT / f"nll_{name}.jsonl")
        if len(nl) == len(ids):
            N[name] = nl
            r["nll_code"] = round(sum(x["nll_sum"] for x in nl.values()) / sum(x["n_tokens"] for x in nl.values()), 5)
        ps = _done(OUT / f"pass_{name}.jsonl")
        if len(ps) == len(ids):
            P[name] = {i: ps[i]["passed"] for i in ids}
            r["pass_at_1"] = round(sum(P[name].values()) / len(ids), 4)
            r["passed"] = sum(P[name].values())
            r["gen_seconds_mean"] = round(sum(ps[i]["gen_seconds"] for i in ids) / len(ids), 2)
            r["syntax_error_rate"] = round(sum("SyntaxError" in (ps[i]["error"] or "") for i in ids) / len(ids), 4)
            r["timeouts"] = sum(ps[i]["error"] == "timeout" for i in ids)
        mp = OUT / f"meta_pass_{name}.json"
        if mp.exists():
            r["peak_vram_gb"] = json.loads(mp.read_text())["memory"].get("peak_allocated_gb")
        res["configs"][name] = r
    for x, ref in [("D", "R"), ("B", "R"), ("T_05", "R"), ("T_10", "R"), ("RB_05", "R"), ("D", "T_10"),
                   ("D", "T_05"), ("T_10", "RB_05"), ("T_05", "RB_05"), ("D", "B")]:
        if x in P and ref in P:
            c = paired_bootstrap([P[ref][i] for i in ids], [P[x][i] for i in ids])
            if x in N and ref in N:
                c["d_nll_code"] = round(res["configs"][x]["nll_code"] - res["configs"][ref]["nll_code"], 5)
            res["comparisons"][f"{x} vs {ref}"] = c
    if "R" in P and "D" in P:
        res["subskill_gap_R_vs_D"] = gap_by_subskill(data, P["R"], P["D"])
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["configs"], indent=1))
    for k, c in res["comparisons"].items():
        print(f"{k:14} {c['diff']*100:+.1f} pp [{c['ci_low']*100:+.1f}, {c['ci_high']*100:+.1f}] "
              f"+{c['wrong_to_right']}/-{c['right_to_wrong']}  ΔNLL {c.get('d_nll_code', float('nan')):+.4f}")
    for s, g in res.get("subskill_gap_R_vs_D", {}).items():
        print(f"  {s:14} n={g['n']:3} R={g['R']:3} D={g['D']:3} gap={g['gap_pp']:+.1f} pp")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="all", choices=["weights", "nll", "pass", "report", "all"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    a = ap.parse_args()
    OUT = Path(a.out)
    if a.step in ("weights", "all"):
        weights()
    for ph in ("nll", "pass"):
        if a.step in (ph, "all"):
            run(ph, a.only, a.batch_size, a.dtype)
    if a.step in ("report", "all"):
        report()
