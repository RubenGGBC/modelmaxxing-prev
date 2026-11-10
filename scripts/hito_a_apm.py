"""Hito A (APM): receptor original vs recorte fijo (IM) vs APM guiado por activaciones.

Uso:  PYTHONPATH=src python scripts/hito_a_apm.py CONFIG.yaml FASE [--limit N]

Fases (en orden; todas escriben en `out` de la configuración y son reanudables):
  resources   comprueba dispositivo, RAM, disco, sandbox y modelos; FALLA si no alcanza (no descarga nada)
  profile     estadísticas de activación del donante, capa a capa (calibración = partición `calibration`)
  slice       construye los slices IM y APM con la forma del receptor (directorios nuevos + hashes)
  slice_nll   NLL de cada slice por sí solo (μ=1) y del receptor, en la partición `probe` (diagnóstico)
  dev         R (μ=0) y cada método × μ en la partición de desarrollo: pass@1 + NLL por ejemplo
  lock        elige μ por método con la regla pre-registrada y lo congela (sha256) ANTES del test
  test        evalúa R y los μ congelados en la partición de test (exige `lock`)
  report      tablas agregadas y bootstrap pareado por problema
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from model_maxxing.evaluation.metrics import paired_bootstrap
from model_maxxing.evaluation.runner import (STOP, best_device, build_prompt, evaluate_mbpp_batched, run_tests,
                                            sandbox_description)
from model_maxxing.evaluation.splits import check_disjoint, ids_hash, load_split
from model_maxxing.models.loading import dtype_report, materialize, neutral_generation_config
from model_maxxing.transfer.apm import (ActivationStats, ArchSpec, apm_recipe, build_slice, check_compat, select_apm,
                                        select_im, sha256_file)
from model_maxxing.transfer.apm_profile import device_memory, profile_donor, write_stats

DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# --------------------------------------------------------------------------- utilidades

def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def done_rows(path: Path) -> dict:
    if not path.exists():
        return {}
    return {json.loads(l)["task_id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}


def environment() -> dict:
    import transformers
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "device": best_device(),
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "sandbox": sandbox_description(), "git_commit": commit, "git_dirty": dirty}


def data_part(spec: dict, limit: int | None = None) -> list[dict]:
    rows = load_split(spec["split"])
    off, n = spec.get("offset", 0), spec.get("n")
    rows = rows[off:off + n] if n else rows[off:]
    return rows[:limit] if limit else rows


def check_partitions(cfg: dict) -> dict:
    """Las partes usadas no se solapan entre sí ni con el conjunto quemado de v0.3."""
    parts = {k: load_split(k) for k in ("diagnostic", "selection", "final_mbpp")}
    check_disjoint(parts)
    used = {k: data_part(cfg[k]) for k in ("calibration", "probe", "dev", "test") if cfg.get(k)}
    names = list(used)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = {r["task_id"] for r in used[a]} & {r["task_id"] for r in used[b]}
            if inter:
                raise SystemExit(f"{a} y {b} comparten {len(inter)} problemas: {sorted(inter)[:5]}")
    return {k: {"split": cfg[k]["split"], "n": len(v), "ids_sha256_16": ids_hash(v)} for k, v in used.items()}


def receiver_config(cfg: dict) -> dict:
    if cfg.get("receiver_arch_override"):
        base = json.loads((Path(cfg["donor"]["path"]) / "config.json").read_text())
        return {**base, **cfg["receiver_arch_override"]}
    return json.loads((Path(cfg["receiver"]["path"]) / "config.json").read_text())


def tokenizer_path(cfg: dict) -> str:
    return cfg["receiver"]["path"] if cfg.get("receiver") else cfg["donor"]["path"]


# --------------------------------------------------------------------------- fases

def phase_resources(cfg: dict, out: Path) -> None:
    import psutil
    need = cfg["resources"]
    free_disk = shutil.disk_usage(".").free / 1e9
    vm = psutil.virtual_memory()
    rep = {"env": environment(), "ram_total_gb": round(vm.total / 1e9, 1), "ram_available_gb": round(vm.available / 1e9, 1),
           "disk_free_gb": round(free_disk, 1), "models": {}, "problems": []}
    for role in ("donor", "receiver"):
        m = cfg.get(role)
        if not m:
            continue
        p = Path(m["path"])
        present = p.exists() and any(p.glob("*.safetensors"))
        rep["models"][role] = {"path": str(p), "present": present, "expected_gb": m.get("size_gb")}
        if not present:          # aviso, no error: se cuenta en el disco necesario y se descarga después
            rep.setdefault("warnings", []).append(
                f"falta {role} en {p} ({m.get('size_gb')} GB; repo {m.get('repo')}@{m.get('revision')})")
    missing_gb = sum(m["expected_gb"] or 0 for m in rep["models"].values() if not m["present"])
    disk_needed = missing_gb + need["work_disk_gb"] + need["disk_margin_gb"]
    rep["disk_needed_gb"] = round(disk_needed, 1)
    if free_disk < disk_needed:
        rep["problems"].append(f"disco libre {free_disk:.1f} GB < {disk_needed:.1f} GB necesarios")
    if vm.total / 1e9 < need["min_ram_gb"]:
        rep["problems"].append(f"RAM {vm.total / 1e9:.1f} GB < {need['min_ram_gb']} GB")
    ok, _ = run_tests("x = 1", {"test_list": ["assert x == 1"]})
    bad, _ = run_tests("x = 1", {"test_list": ["assert x == 2"]})
    rep["sandbox_ok"] = bool(ok and not bad)
    if not rep["sandbox_ok"]:
        rep["problems"].append("sandbox de ejecución de código no operativo")
    rep["partitions"] = check_partitions(cfg)
    jdump(out / "resources.json", rep)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if rep["problems"]:
        raise SystemExit(2)


def phase_profile(cfg: dict, out: Path) -> None:
    from transformers import AutoTokenizer
    c = cfg["calibration"]
    tok = AutoTokenizer.from_pretrained(cfg["donor"]["path"])
    rows = data_part(c)
    exs = []
    for ex in rows:
        prompt = build_prompt(ex)
        ids = tok(prompt).input_ids
        if c["scope"] == "prompt":
            mask = [True] * len(ids)
        elif c["scope"] == "prompt+reference":              # variante: solución de referencia por teacher forcing
            ids = tok(prompt + ex["code"].rstrip() + "\n" + STOP).input_ids
            mask = [True] * len(ids)
        else:
            raise ValueError(c["scope"])
        exs.append((ids[:c["max_tokens"]], mask[:c["max_tokens"]]))
    dev = best_device()
    print(f"perfilando {cfg['donor']['path']} en {dev}: {len(exs)} ejemplos, {sum(len(i) for i, _ in exs)} tokens")
    stats, cost = profile_donor(cfg["donor"]["path"], exs, dev, DT[cfg["profile_dtype"]])
    rec = write_stats(stats, cost, out / "profile", {"calibration": {**c, "ids_sha256_16": ids_hash(rows)},
                                                      "donor": cfg["donor"], "env": environment()})
    print("perfil:", rec["stats_sha256"][:16], f"{cost['total_seconds']} s", cost["memory"])


def phase_slice(cfg: dict, out: Path) -> None:
    donor = ArchSpec.from_path(cfg["donor"]["path"])
    rcfg = receiver_config(cfg)
    recip = ArchSpec.from_config(rcfg)
    errs = check_compat(donor, recip)
    if errs:
        raise SystemExit("incompatible: " + "; ".join(errs))
    stats = ActivationStats.load(out / "profile" / "activation_stats.pt")
    for method in cfg["methods"]:
        d = out / "slices" / method
        if (d / "manifest.json").exists():
            print("ya hecho:", method)
            continue
        sel = select_im(donor, recip) if method == "IM" else select_apm(stats, donor, recip)
        t0 = time.time()
        man = build_slice(cfg["donor"]["path"], rcfg, sel, d)
        man["seconds"] = round(time.time() - t0, 1)
        jdump(d / "manifest.json", man)
        print(method, man["selection_digest"], man["sha256"]["model.safetensors"][:16], f"{man['seconds']} s")
    ov = overlap(out)
    jdump(out / "slices" / "overlap_IM_APM.json", ov)
    print("solapamiento IM∩APM:", ov)


def overlap(out: Path) -> dict:
    s = {m: json.loads((out / "slices" / m / "selection.json").read_text()) for m in ("IM", "APM")
         if (out / "slices" / m / "selection.json").exists()}
    if len(s) < 2:
        return {}
    frac = lambda a, b: len(set(a) & set(b)) / max(len(a), 1)
    im, ap = s["IM"], s["APM"]
    L = len(ap["layers"])
    return {"layers": frac(im["layers"], ap["layers"]), "hidden": frac(im["hidden"], ap["hidden"]),
            "q_heads_mean": sum(frac(im["q_heads"][i], ap["q_heads"][i]) for i in range(L)) / L,
            "kv_heads_mean": sum(frac(im["kv_heads"][i], ap["kv_heads"][i]) for i in range(L)) / L,
            "mlp_mean": sum(frac(im["mlp"][i], ap["mlp"][i]) for i in range(L)) / L}


@torch.no_grad()
def solution_nll(model, tok, dev, ex: dict) -> tuple[float, int]:
    """NLL (suma) de la solución de referencia + cierre de bloque, dada la consigna (igual que select_models)."""
    prompt = build_prompt(ex)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + ex["code"].rstrip() + "\n" + STOP, return_tensors="pt").input_ids
    n_p = p_ids.shape[1]
    if not torch.equal(full[0, :n_p], p_ids[0]):
        n_p -= 1
    ids = full[:, :1024].to(dev)
    lg = model(ids).logits.float()
    nll = torch.nn.functional.cross_entropy(lg[0, n_p - 1:-1], ids[0, n_p:], reduction="sum").item()
    return nll, int(ids.shape[1] - n_p)


def load_causal_lm(path: str, dtype, dev: str):
    """En CUDA carga directamente en la GPU (device_map): evita una copia FP32 completa en RAM, importante
    cuando dos procesos (uno por GPU) cargan a la vez en una VM con poca RAM (Kaggle)."""
    from transformers import AutoModelForCausalLM
    if dev.startswith("cuda"):
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=dev).eval()
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()


def load_receiver_skeleton(path: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = best_device()
    tok = AutoTokenizer.from_pretrained(path)
    model = load_causal_lm(path, dtype, dev)
    model.generation_config = neutral_generation_config(tok)
    return model, tok, dev


def phase_slice_nll(cfg: dict, out: Path, limit: int | None) -> None:
    rows = data_part(cfg["probe"], limit)
    targets = {m: out / "slices" / m for m in cfg["methods"]}
    if cfg.get("receiver"):
        targets = {"R": Path(cfg["receiver"]["path"]), **targets}
    res = {}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path(cfg))
    dev = best_device()
    for name, path in targets.items():
        model = load_causal_lm(str(path), DT[cfg["eval"]["dtype"]], dev)
        s = n = 0
        for ex in rows:
            a, b = solution_nll(model, tok, dev, ex)
            s, n = s + a, n + b
        res[name] = {"nll_per_token": round(s / n, 4), "tokens": n}
        print(name, res[name])
        del model
    jdump(out / "slice_nll.json", {"probe": {**cfg["probe"], "n": len(rows), "ids_sha256_16": ids_hash(rows)},
                                   "results": res, "note": "slice solo (μ=1): diagnóstico, no es un brazo"})


def variants_for(cfg: dict, stage: str) -> dict[str, tuple[str, float]]:
    """nombre → (método, μ). R siempre es μ=0."""
    v = {"R": ("R", 0.0)}
    if stage == "dev":
        for m in cfg["methods"]:
            for mu in cfg["mu_grid"]:
                v[f"{m}_mu{mu:g}"] = (m, mu)
    else:
        lock = json.loads((Path(cfg["out"]) / "locked_choice.json").read_text())
        for m, mu in lock["chosen_mu"].items():
            v[f"{m}_mu{mu:g}"] = (m, mu)
    return v


def evaluate_stage(cfg: dict, out: Path, stage: str, limit: int | None, only: list[str] | None = None) -> None:
    part = cfg["dev"] if stage == "dev" else cfg["test"]
    rows = data_part(part, limit)
    ok, _ = run_tests("x = 1", {"test_list": ["assert x == 1"]})
    bad, _ = run_tests("x = 1", {"test_list": ["assert x == 2"]})
    if not ok or bad:
        raise SystemExit("sandbox no operativo: no se ejecuta código generado")
    E = cfg["eval"]
    model, tok, dev = load_receiver_skeleton(cfg["receiver"]["path"], DT[E["dtype"]])
    sd = out / stage
    jdump(sd / "data_manifest.json", {**part, "n": len(rows), "ids_sha256_16": ids_hash(rows), "limit": limit})
    variants = variants_for(cfg, stage)
    if only:
        unknown = set(only) - set(variants)
        if unknown:
            raise SystemExit(f"--only con variantes desconocidas {sorted(unknown)}; existen {sorted(variants)}")
        variants = {k: v for k, v in variants.items() if k in only}
    for name, (method, mu) in variants.items():
        pf, nf, mf = sd / f"pass_{name}.jsonl", sd / f"nll_{name}.jsonl", sd / f"meta_{name}.json"
        todo_p = [e for e in rows if e["task_id"] not in done_rows(pf)]
        todo_n = [e for e in rows if e["task_id"] not in done_rows(nf)]
        if not todo_p and not todo_n:
            print("ya hecho:", stage, name)
            continue
        slice_dir = str(out / "slices" / method) if method != "R" else None
        rec = apm_recipe(cfg["receiver"]["path"], slice_dir, mu) if slice_dir else apm_recipe(cfg["receiver"]["path"], "", 0.0)
        t0 = time.time()
        info = materialize(model, rec)
        t_merge = time.time() - t0
        t1 = time.time()
        with open(nf, "a") as fh:
            for ex in todo_n:
                s, n = solution_nll(model, tok, dev, ex)
                fh.write(json.dumps({"task_id": ex["task_id"], "nll_sum": s, "n_tokens": n}) + "\n")
        t_nll = time.time() - t1

        def save(rs, pf=pf):
            with open(pf, "a") as fh:
                for r in rs:
                    fh.write(json.dumps(r) + "\n")
        t2 = time.time()
        if todo_p:
            evaluate_mbpp_batched(model, tok, dev, todo_p, tag=name, batch_size=E["batch_size"],
                                  max_new_tokens=E["max_new_tokens"], on_rows=save)
        meta = {"name": name, "method": method, "mu": mu, "recipe": rec.describe(),
                "slice_sha256": (json.loads((Path(slice_dir) / "manifest.json").read_text())["sha256"]["model.safetensors"]
                                 if slice_dir else None), **info,
                "seconds": {"merge": round(t_merge, 1), "nll": round(t_nll, 1), "pass": round(time.time() - t2, 1)},
                "memory": device_memory(dev), "dtypes": dtype_report(model), "eval": E, "env": environment()}
        jdump(mf, meta)
        print(stage, name, meta["seconds"], meta["memory"], flush=True)


def summarize(sd: Path, rows_ids: list) -> dict:
    out = {}
    for pf in sorted(sd.glob("pass_*.jsonl")):
        name = pf.stem[5:]
        P, N = done_rows(pf), done_rows(sd / f"nll_{name}.jsonl")
        if set(P) != set(rows_ids):
            continue
        out[name] = {"passed": sum(P[i]["passed"] for i in rows_ids), "n": len(rows_ids),
                     "pass_at_1": round(sum(P[i]["passed"] for i in rows_ids) / len(rows_ids), 4),
                     "nll_code": round(sum(N[i]["nll_sum"] for i in rows_ids) / sum(N[i]["n_tokens"] for i in rows_ids), 5)
                     if set(N) == set(rows_ids) else None,
                     "syntax_errors": sum("SyntaxError" in (P[i]["error"] or "") for i in rows_ids),
                     "timeouts": sum(P[i]["error"] == "timeout" for i in rows_ids),
                     "_vec": [bool(P[i]["passed"]) for i in rows_ids]}
    return out


def phase_lock(cfg: dict, out: Path) -> None:
    lf = out / "locked_choice.json"
    if lf.exists():
        raise SystemExit(f"{lf} ya existe: el μ ya está congelado")
    ids = sorted(done_rows(out / "dev" / f"pass_R.jsonl"))
    S = summarize(out / "dev", ids)
    chosen = {}
    for m in cfg["methods"]:
        cands = [(S[f"{m}_mu{mu:g}"]["pass_at_1"], -(S[f"{m}_mu{mu:g}"]["nll_code"] or 9e9), -mu, mu)
                 for mu in cfg["mu_grid"] if f"{m}_mu{mu:g}" in S]
        if len(cands) != len(cfg["mu_grid"]):
            raise SystemExit(f"faltan evaluaciones de dev para {m}")
        chosen[m] = max(cands)[3]           # regla: mejor pass@1; empate → menor NLL; empate → menor μ
    body = {"rule": cfg["mu_selection_rule"], "chosen_mu": chosen, "dev_n": len(ids),
            "dev_summary": {k: {kk: vv for kk, vv in v.items() if kk != "_vec"} for k, v in S.items()},
            "slices": {m: json.loads((out / "slices" / m / "manifest.json").read_text())["sha256"] for m in cfg["methods"]},
            "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(), "env": environment()}
    jdump(lf, body)
    print("congelado:", chosen, "sha256", sha256_file(lf)[:16])


def phase_report(cfg: dict, out: Path) -> None:
    rep = {}
    for stage in ("dev", "test"):
        sd = out / stage
        if not (sd / "pass_R.jsonl").exists():
            continue
        ids = sorted(done_rows(sd / "pass_R.jsonl"))
        S = summarize(sd, ids)
        comps = {}
        for name in S:
            if name != "R":
                comps[f"{name} vs R"] = paired_bootstrap(S["R"]["_vec"], S[name]["_vec"])
        ims = [n for n in S if n.startswith("IM_")]
        apms = [n for n in S if n.startswith("APM_")]
        if stage == "test" and ims and apms:
            comps[f"{apms[0]} vs {ims[0]}"] = paired_bootstrap(S[ims[0]]["_vec"], S[apms[0]]["_vec"])
        rep[stage] = {"n": len(ids), "arms": {k: {kk: vv for kk, vv in v.items() if kk != "_vec"} for k, v in S.items()},
                      "paired_bootstrap": comps}
    jdump(out / "report.json", rep)
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("phase", choices=["resources", "profile", "slice", "slice_nll", "dev", "lock", "test", "report"])
    ap.add_argument("--limit", type=int, default=None, help="solo para pruebas técnicas: limita ejemplos")
    ap.add_argument("--only", nargs="+", default=None,
                    help="dev/test: solo estas variantes (p. ej. para repartirlas entre GPUs con CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--list-variants", action="store_true", help="dev/test: imprime las variantes y sale")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg["_config_path"] = a.config
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    if a.phase == "resources":
        phase_resources(cfg, out)
    elif a.phase == "profile":
        phase_profile(cfg, out)
    elif a.phase == "slice":
        phase_slice(cfg, out)
    elif a.phase == "slice_nll":
        phase_slice_nll(cfg, out, a.limit)
    elif a.phase in ("dev", "test"):
        if a.phase == "test" and not (out / "locked_choice.json").exists():
            raise SystemExit("test exige locked_choice.json (fase lock) antes de evaluar")
        if a.list_variants:
            print(" ".join(variants_for(cfg, a.phase)))
            return
        evaluate_stage(cfg, out, a.phase, a.limit, a.only)
    elif a.phase == "lock":
        phase_lock(cfg, out)
    else:
        phase_report(cfg, out)


if __name__ == "__main__":
    main()
