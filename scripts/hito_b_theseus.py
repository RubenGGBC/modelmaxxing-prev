"""Hito B (Theseus): traslado del task vector Coder-7B − Base-7B a Qwen2.5-1.5B. Ver docs/HITO_B_THESEUS_PLAN.md.

Uso:  PYTHONPATH=src python scripts/hito_b_theseus.py CONFIG.yaml FASE [--only V ...] [--list-variants] [--limit N]

Fases (en orden, reanudables):
  resources  disco/RAM/sandbox/particiones (no descarga nada)
  align      alineamiento capa a capa A↔B y deltas TH / RAND / NAIVE  (+ diagnóstico de similitud)
  slices     slices de referencia del Hito A (APM y APMRAND) desde las estadísticas guardadas
  dev_nll    NLL de código en dev de R, todos los brazos × α y los brazos fijos
  pick       por brazo con α, los `nll_top_k` α de mejor NLL en dev (se congela)
  dev_pass   pass@1 en dev de R, los α elegidos y los brazos fijos
  lock       α final por brazo (pass@1 → NLL → α menor) congelado con hashes ANTES del test
  test       R + α congelados + brazos fijos en cada partición de test (MBPP final y HumanEval+)
  report     tablas por partición y combinadas, bootstrap pareado frente a R y frente a APM
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from hito_a_apm import data_part, done_rows, environment, jdump, load_receiver_skeleton  # noqa: E402

from model_maxxing.evaluation import humaneval  # noqa: E402
from model_maxxing.evaluation.metrics import paired_bootstrap  # noqa: E402
from model_maxxing.evaluation.runner import STOP, best_device, build_prompt, evaluate_mbpp_batched, run_tests  # noqa: E402
from model_maxxing.evaluation.splits import check_disjoint, ids_hash, load_split  # noqa: E402
from model_maxxing.models.loading import dtype_report  # noqa: E402
from model_maxxing.transfer.apm import (ActivationStats, ArchSpec, build_slice, random_selection,  # noqa: E402
                                        select_apm, sha256_file)
from model_maxxing.transfer.apm_profile import device_memory  # noqa: E402
from model_maxxing.transfer.theseus import QK_KEYS, align_and_transport, materialize_affine  # noqa: E402

DT = {"float32": torch.float32, "bfloat16": torch.bfloat16}


# --------------------------------------------------------------------------- variantes

def variant_spec(cfg: dict, name: str) -> dict:
    """nombre → argumentos de materialize_affine (delta_dir, scale, alpha, exclude)."""
    out = Path(cfg["out"])
    if name == "R":
        return {"delta_dir": None, "scale": 1.0, "alpha": 0.0, "exclude": ()}
    arm, _, val = name.rpartition("_")
    x = float(val[1:] if val.startswith("a") else val[2:])
    if val.startswith("a"):                                          # brazos Theseus: θ_B + α·τ_B
        d = {"TH": "TH", "THnoQK": "TH", "NAIVE": "NAIVE", "RAND": "RAND"}[arm]
        return {"delta_dir": str(out / "deltas" / d), "scale": 1.0, "alpha": x,
                "exclude": QK_KEYS if arm == "THnoQK" else ()}
    if arm in ("APM", "APMRAND"):                                    # (1−μ)θ_B + μ·slice
        return {"delta_dir": str(out / "slices" / arm), "scale": 1 - x, "alpha": x, "exclude": ()}
    if arm == "SHRINK":                                              # (1−μ)θ_B, sin donante
        return {"delta_dir": None, "scale": 1 - x, "alpha": 0.0, "exclude": ()}
    raise ValueError(name)


def alpha_variants(cfg: dict, alphas: dict[str, list[float]] | None = None) -> list[str]:
    alphas = alphas or {a: cfg["alpha_grid"] for a in cfg["alpha_arms"]}
    return [f"{a}_a{x:g}" for a in cfg["alpha_arms"] for x in alphas[a]]


def variants_for(cfg: dict, stage: str) -> list[str]:
    out = Path(cfg["out"])
    fixed = list(cfg["fixed_arms"])
    if stage == "dev_nll":
        return ["R"] + alpha_variants(cfg) + fixed
    if stage == "dev_pass":
        pick = json.loads((out / "dev_pick.json").read_text())["picked"]
        return ["R"] + alpha_variants(cfg, pick) + fixed
    if stage == "test":
        lock = json.loads((out / "locked_choice.json").read_text())["chosen_alpha"]
        return ["R"] + [f"{a}_a{x:g}" for a, x in lock.items()] + fixed
    raise ValueError(stage)


# --------------------------------------------------------------------------- tareas

def task_of(split: str):
    return humaneval if split == "final_humaneval" else None


@torch.no_grad()
def solution_nll(model, tok, dev, ex: dict, task) -> tuple[float, int]:
    """NLL (suma) de la solución de referencia dada la consigna, con el formato de la tarea."""
    prompt = task.build_prompt(ex) if task else build_prompt(ex)
    target = ex["code"] if task else ex["code"].rstrip() + "\n" + STOP
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + target, return_tensors="pt").input_ids
    n_p = p_ids.shape[1]
    if not torch.equal(full[0, :n_p], p_ids[0]):
        n_p -= 1
    ids = full[:, :2048].to(dev)
    lg = model(ids).logits.float()
    nll = torch.nn.functional.cross_entropy(lg[0, n_p - 1:-1], ids[0, n_p:], reduction="sum").item()
    return nll, int(ids.shape[1] - n_p)


def splits_for(cfg: dict, stage: str) -> list[dict]:
    return [cfg["dev"]] if stage.startswith("dev") else cfg["test"]


# --------------------------------------------------------------------------- fases

def phase_resources(cfg: dict, out: Path) -> None:
    import psutil
    free = shutil.disk_usage(".").free / 1e9
    vm = psutil.virtual_memory()
    rep = {"env": environment(), "ram_total_gb": round(vm.total / 1e9, 1), "disk_free_gb": round(free, 1),
           "models": {}, "problems": [], "warnings": []}
    missing = 0.0
    for role, m in cfg["models"].items():
        present = Path(m["path"]).exists() and any(Path(m["path"]).glob("*.safetensors"))
        rep["models"][role] = {"path": m["path"], "present": present}
        if not present:
            missing += m["size_gb"]
            rep["warnings"].append(f"falta {role}: {m['repo']}@{m['revision']} ({m['size_gb']} GB)")
    need = missing + cfg["resources"]["work_disk_gb"] + cfg["resources"]["disk_margin_gb"]
    rep["disk_needed_gb"] = round(need, 1)
    if free < need:
        rep["problems"].append(f"disco libre {free:.1f} GB < {need:.1f} GB")
    if vm.total / 1e9 < cfg["resources"]["min_ram_gb"]:
        rep["problems"].append(f"RAM {vm.total / 1e9:.1f} GB < {cfg['resources']['min_ram_gb']} GB")
    ok, _ = run_tests("x = 1", {"test_list": ["assert x == 1"]})
    bad, _ = run_tests("x = 1", {"test_list": ["assert x == 2"]})
    he_ok = humaneval.run({"prompt": "def f(x):\n", "entry_point": "f",
                           "test": "def check(c):\n    assert c(1) == 2\n"}, "    return x + 1\n")[0]
    rep["sandbox_ok"] = bool(ok and not bad and he_ok)
    if not rep["sandbox_ok"]:
        rep["problems"].append("sandbox no operativo (MBPP o HumanEval)")
    parts = {k: load_split(k) for k in ("diagnostic", "selection", "final_mbpp")}
    check_disjoint(parts)
    rep["partitions"] = {s["split"]: {"n": len(data_part(s)), "ids_sha256_16": ids_hash(data_part(s))}
                         for s in [cfg["alignment"], cfg["dev"], *cfg["test"]]}
    jdump(out / "resources.json", rep)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if rep["problems"]:
        raise SystemExit(2)


def phase_align(cfg: dict, out: Path) -> None:
    from transformers import AutoTokenizer
    M, c = cfg["models"], cfg["alignment"]
    if (out / "deltas" / "TH" / "manifest.json").exists():
        print("ya hecho: align")
        return
    tb = AutoTokenizer.from_pretrained(M["receiver"]["path"])
    ta = AutoTokenizer.from_pretrained(M["base_a"]["path"])
    rows = data_part(c)
    texts = [build_prompt(e) + e["code"].rstrip() + "\n" + STOP if c["scope"] == "prompt+reference" else build_prompt(e)
             for e in rows]
    ex = [tb(t).input_ids[:c["max_tokens"]] for t in texts]
    if ex != [ta(t).input_ids[:c["max_tokens"]] for t in texts]:
        raise SystemExit("los tokenizers de A y B no tokenizan igual: el emparejamiento 1:1 de tokens no vale")
    dev = best_device()
    print(f"alineando en {dev}: {len(ex)} ejemplos, {sum(map(len, ex))} tokens")
    t0 = time.time()
    res = align_and_transport(M["base_a"]["path"], M["ft_a"]["path"], M["receiver"]["path"], ex, out / "deltas", dev,
                              DT[c["dtype"]], rand_seed=c["rand_seed"])
    jdump(out / "deltas" / "align_meta.json", {"alignment": {**c, "n": len(rows), "ids_sha256_16": ids_hash(rows),
                                                             "tokens": sum(map(len, ex))},
                                               "seconds": round(time.time() - t0, 1), "memory": device_memory(dev),
                                               "env": environment(), "models": M})
    print("align:", res["total_seconds"], "s", device_memory(dev))


def phase_slices(cfg: dict, out: Path) -> None:
    ref = cfg["apm_reference"]
    if sha256_file(ref["stats"]) != ref["stats_sha256"]:
        raise SystemExit("las estadísticas del Hito A no coinciden con el hash registrado")
    base = cfg["models"]["base_a"]["path"]
    rcfg = json.loads((Path(cfg["models"]["receiver"]["path"]) / "config.json").read_text())
    donor, recip = ArchSpec.from_path(base), ArchSpec.from_config(rcfg)
    sels = {"APM": select_apm(ActivationStats.load(ref["stats"]), donor, recip),
            "APMRAND": random_selection(donor, recip, ref["rand_seed"])}
    for name, sel in sels.items():
        d = out / "slices" / name
        if (d / "manifest.json").exists():
            print("ya hecho:", name)
            continue
        man = build_slice(base, rcfg, sel, d)
        print(name, man["selection_digest"], man["sha256"]["model.safetensors"][:16])
    a = json.loads((out / "slices" / "APM" / "manifest.json").read_text())
    if ref.get("apm_slice_sha256") and a["sha256"]["model.safetensors"] != ref["apm_slice_sha256"]:
        raise SystemExit("el slice APM reconstruido no coincide con el del Hito A")


def evaluate(cfg: dict, out: Path, stage: str, only: list[str] | None, limit: int | None) -> None:
    names = variants_for(cfg, stage)
    if only:
        unknown = set(only) - set(names)
        if unknown:
            raise SystemExit(f"--only desconocidas {sorted(unknown)}; existen {names}")
        names = [n for n in names if n in only]
    E = cfg["eval"]
    do_pass, do_nll = stage in ("dev_pass", "test"), stage in ("dev_nll", "test")
    if do_pass:
        ok, _ = run_tests("x = 1", {"test_list": ["assert x == 1"]})
        if not ok:
            raise SystemExit("sandbox no operativo")
    model, tok, dev = load_receiver_skeleton(cfg["models"]["receiver"]["path"], DT[E["dtype"]])
    parts = {s["split"]: data_part(s, limit) for s in splits_for(cfg, stage)}
    sdir = out / ("dev" if stage.startswith("dev") else "test")
    for split, rows in parts.items():
        jdump(sdir / split / "data_manifest.json", {"split": split, "n": len(rows), "ids_sha256_16": ids_hash(rows),
                                                    "limit": limit})
    for name in names:
        todo = {}
        for split, rows in parts.items():
            pf, nf = sdir / split / f"pass_{name}.jsonl", sdir / split / f"nll_{name}.jsonl"
            todo[split] = ([e for e in rows if e["task_id"] not in done_rows(pf)] if do_pass else [],
                           [e for e in rows if e["task_id"] not in done_rows(nf)] if do_nll else [])
        if not any(a or b for a, b in todo.values()):
            print("ya hecho:", stage, name)
            continue
        spec = variant_spec(cfg, name)
        t0 = time.time()
        info = materialize_affine(model, cfg["models"]["receiver"]["path"], **spec)
        secs = {"merge": round(time.time() - t0, 1)}
        for split, (tp, tn) in todo.items():
            task = task_of(split)
            t1 = time.time()
            with open(sdir / split / f"nll_{name}.jsonl", "a") as fh:
                for ex in tn:
                    s, n = solution_nll(model, tok, dev, ex, task)
                    fh.write(json.dumps({"task_id": ex["task_id"], "nll_sum": s, "n_tokens": n}) + "\n")
            secs[f"nll_{split}"] = round(time.time() - t1, 1)
            if tp:
                def save(rs, f=sdir / split / f"pass_{name}.jsonl"):
                    with open(f, "a") as fh:
                        for r in rs:
                            fh.write(json.dumps(r) + "\n")
                t2 = time.time()
                evaluate_mbpp_batched(model, tok, dev, tp, tag=name, batch_size=E["batch_size"],
                                      max_new_tokens=E["max_new_tokens"], on_rows=save,
                                      max_batch_tokens=E.get("max_batch_tokens"), task=task)
                secs[f"pass_{split}"] = round(time.time() - t2, 1)
        meta = {"name": name, "spec": {**spec, "exclude": list(spec["exclude"])}, **info, "seconds": secs,
                "memory": device_memory(dev), "dtypes": dtype_report(model), "eval": E, "env": environment()}
        jdump(sdir / f"meta_{name}.json", meta)
        print(stage, name, secs, meta["memory"], flush=True)


def _nll(sd: Path, split: str, name: str, ids: list) -> float | None:
    N = done_rows(sd / split / f"nll_{name}.jsonl")
    if set(N) != set(ids):
        return None
    return sum(N[i]["nll_sum"] for i in ids) / sum(N[i]["n_tokens"] for i in ids)


def phase_pick(cfg: dict, out: Path) -> None:
    pf = out / "dev_pick.json"
    if pf.exists():
        raise SystemExit(f"{pf} ya existe (congelado)")
    split = cfg["dev"]["split"]
    ids = sorted(e["task_id"] for e in data_part(cfg["dev"]))
    nll = {n: _nll(out / "dev", split, n, ids) for n in variants_for(cfg, "dev_nll")}
    if any(v is None for v in nll.values()):
        raise SystemExit(f"faltan NLL de dev: {[k for k, v in nll.items() if v is None]}")
    picked = {}
    for a in cfg["alpha_arms"]:
        ranked = sorted(cfg["alpha_grid"], key=lambda x: (nll[f"{a}_a{x:g}"], x))
        picked[a] = sorted(ranked[:cfg["nll_top_k"]])
    jdump(pf, {"rule": f"por brazo, los {cfg['nll_top_k']} α con menor NLL de código en dev (empate → α menor)",
               "picked": picked, "dev_nll": {k: round(v, 5) for k, v in nll.items()}})
    print("elegidos para pass@1:", picked)


def phase_lock(cfg: dict, out: Path) -> None:
    lf = out / "locked_choice.json"
    if lf.exists():
        raise SystemExit(f"{lf} ya existe (congelado)")
    split = cfg["dev"]["split"]
    ids = sorted(e["task_id"] for e in data_part(cfg["dev"]))
    pick = json.loads((out / "dev_pick.json").read_text())["picked"]
    chosen, table = {}, {}
    for a in cfg["alpha_arms"]:
        cands = []
        for x in pick[a]:
            n = f"{a}_a{x:g}"
            P = done_rows(out / "dev" / split / f"pass_{n}.jsonl")
            if set(P) != set(ids):
                raise SystemExit(f"faltan pass@1 de dev para {n}")
            p = sum(P[i]["passed"] for i in ids)
            table[n] = {"passed": p, "nll": round(_nll(out / "dev", split, n, ids), 5)}
            cands.append((p, -table[n]["nll"], -x, x))
        chosen[a] = max(cands)[3]           # pass@1 → menor NLL → menor α
    for n in ["R", *cfg["fixed_arms"]]:
        P = done_rows(out / "dev" / split / f"pass_{n}.jsonl")
        table[n] = {"passed": sum(P[i]["passed"] for i in ids) if set(P) == set(ids) else None,
                    "nll": round(_nll(out / "dev", split, n, ids) or float("nan"), 5)}
    body = {"rule": cfg["selection_rule"], "chosen_alpha": chosen, "dev_n": len(ids), "dev_summary": table,
            "deltas_sha256": {a: json.loads((out / "deltas" / a / "manifest.json").read_text())["sha256"]
                              for a in ("TH", "RAND", "NAIVE")},
            "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(), "env": environment()}
    jdump(lf, body)
    print("congelado:", chosen, "sha256", sha256_file(lf)[:16])


def _boot_nll(sd, split, a, b, ids, n=10000, seed=42):
    A, B = done_rows(sd / split / f"nll_{a}.jsonl"), done_rows(sd / split / f"nll_{b}.jsonl")
    sa, ta = np.array([A[i]["nll_sum"] for i in ids]), np.array([A[i]["n_tokens"] for i in ids])
    sb, tb = np.array([B[i]["nll_sum"] for i in ids]), np.array([B[i]["n_tokens"] for i in ids])
    idx = np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))
    d = sb[idx].sum(1) / tb[idx].sum(1) - sa[idx].sum(1) / ta[idx].sum(1)
    return {"d_nll": round(sb.sum() / tb.sum() - sa.sum() / ta.sum(), 5),
            "ci_low": round(float(np.percentile(d, 2.5)), 5), "ci_high": round(float(np.percentile(d, 97.5)), 5)}


def phase_report(cfg: dict, out: Path) -> None:
    rep = {}
    sd = out / "test"
    if not (out / "locked_choice.json").exists():
        raise SystemExit("sin lock no hay informe de test")
    names = variants_for(cfg, "test")
    vec, ids_all = {n: [] for n in names}, {}
    for s in cfg["test"]:
        split = s["split"]
        ids = sorted(e["task_id"] for e in data_part(s))
        ids_all[split] = ids
        arms = {}
        for n in names:
            P = done_rows(sd / split / f"pass_{n}.jsonl")
            if set(P) != set(ids):
                continue
            v = [bool(P[i]["passed"]) for i in ids]
            vec[n] += v
            arms[n] = {"passed": sum(v), "n": len(ids), "pass_at_1": round(sum(v) / len(ids), 4),
                       "nll_code": round(_nll(sd, split, n, ids) or float("nan"), 5)}
        comps = {}
        for ref in ("R", "APM_mu0.02"):
            for n in arms:
                if n != ref and ref in arms:
                    Pr, Pn = done_rows(sd / split / f"pass_{ref}.jsonl"), done_rows(sd / split / f"pass_{n}.jsonl")
                    comps[f"{n} vs {ref}"] = {**paired_bootstrap([Pr[i]["passed"] for i in ids],
                                                                 [Pn[i]["passed"] for i in ids]),
                                              "nll": _boot_nll(sd, split, ref, n, ids)}
        rep[split] = {"arms": arms, "comparisons": comps}
    pooled = {}
    for ref in ("R", "APM_mu0.02"):
        for n in names:
            if n != ref and len(vec[n]) == len(vec[ref]) == sum(map(len, ids_all.values())):
                pooled[f"{n} vs {ref}"] = paired_bootstrap(vec[ref], vec[n])
    rep["pooled"] = {"n": sum(map(len, ids_all.values())),
                     "pass_at_1": {n: round(sum(v) / len(v), 4) for n, v in vec.items() if v},
                     "comparisons": pooled, "criterion": cfg["criterion"]}
    jdump(out / "report.json", rep)
    print(json.dumps(rep["pooled"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("phase", choices=["resources", "align", "slices", "dev_nll", "pick", "dev_pass", "lock", "test",
                                      "report"])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--list-variants", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="solo pruebas técnicas")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg["_config_path"] = a.config
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    if a.phase in ("dev_nll", "dev_pass", "test"):
        if a.phase == "test" and not (out / "locked_choice.json").exists():
            raise SystemExit("test exige locked_choice.json (fase lock)")
        if a.list_variants:
            print(" ".join(variants_for(cfg, a.phase)))
            return
        evaluate(cfg, out, a.phase, a.only, a.limit)
    else:
        {"resources": phase_resources, "align": phase_align, "slices": phase_slices, "pick": phase_pick,
         "lock": phase_lock, "report": phase_report}[a.phase](cfg, out)


if __name__ == "__main__":
    main()
