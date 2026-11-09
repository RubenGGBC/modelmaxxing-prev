"""Prioridad 2 (v0.3): diagnósticos consultables por el cirujano. Salidas pequeñas y estructuradas.
Los datos que ve el agente (split 'diagnóstico') son distintos de los de evaluación final."""
from __future__ import annotations

import json
from collections import Counter

import torch

# ---------------- activaciones ----------------

@torch.no_grad()
def _collect_down_inputs(model, tok, dev, texts: list[str], layers: list[int], max_len: int = 256):
    """Media por canal de |entrada a down_proj| (activación intermedia del MLP) y RMS de su salida."""
    acc_in = {l: None for l in layers}
    acc_out = {l: 0.0 for l in layers}
    cnt = {l: 0 for l in layers}
    hooks = []
    for l in layers:
        mod = model.model.layers[l].mlp.down_proj

        def hook(m, inp, out, l=l):
            x = inp[0].float().reshape(-1, inp[0].shape[-1])
            s = x.abs().sum(0)
            acc_in[l] = s if acc_in[l] is None else acc_in[l] + s
            acc_out[l] += out.float().pow(2).mean().item() * x.shape[0]
            cnt[l] += x.shape[0]
        hooks.append(mod.register_forward_hook(hook))
    try:
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
            model(ids)
    finally:
        for h in hooks:
            h.remove()
    return ({l: (acc_in[l] / cnt[l]).cpu() for l in layers}, {l: (acc_out[l] / cnt[l]) ** 0.5 for l in layers})


def activation_stats(model, tok, dev, texts: list[str], layers: list[int]) -> dict:
    ins, outs = _collect_down_inputs(model, tok, dev, texts, layers)
    return {"layers": {str(l): {"mlp_out_rms": round(outs[l], 4),
                                 "mlp_mid_mean_abs": round(ins[l].mean().item(), 5),
                                 "mlp_mid_p99_abs": round(ins[l].quantile(0.99).item(), 5)} for l in layers}}


def activation_contrast(model, tok, dev, texts_a: list[str], texts_b: list[str], layers: list[int], top_k: int = 8) -> dict:
    """Canales intermedios (entrada de down_proj) más específicos de A frente a B, por capa."""
    ia, _ = _collect_down_inputs(model, tok, dev, texts_a, layers)
    ib, _ = _collect_down_inputs(model, tok, dev, texts_b, layers)
    res = {}
    for l in layers:
        d = ia[l] - ib[l]
        top = d.topk(top_k)
        res[str(l)] = {"channels_A_specific": top.indices.tolist(), "diff": [round(x, 4) for x in top.values.tolist()],
                       "contrast_energy": round(d.norm().item(), 4)}
    return res


def contrast_channel_mask(model, tok, dev, texts_a, texts_b, layer: int, n_channels: int) -> torch.Tensor:
    ia, _ = _collect_down_inputs(model, tok, dev, texts_a, [layer])
    ib, _ = _collect_down_inputs(model, tok, dev, texts_b, [layer])
    d = ia[layer] - ib[layer]
    mask = torch.zeros_like(d)
    mask[d.topk(n_channels).indices] = 1.0
    return mask


# ---------------- direcciones singulares / deltas ----------------

def singular_directions(delta: torch.Tensor, k: int = 6) -> dict:
    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    e = S.pow(2)
    return {"shape": list(delta.shape), "fro": round(delta.float().norm().item(), 4),
            "top_singular_values": [round(x, 4) for x in S[:k].tolist()],
            "cum_energy_top_k": [round(x, 4) for x in (e.cumsum(0)[:k] / e.sum()).tolist()],
            "stable_rank": round((e.sum() / e[0]).item(), 2)}


def delta_overview(base: dict, donor: dict, pattern: str = "down_proj") -> list[dict]:
    """Un renglón por tensor: tamaño relativo del delta (para elegir dónde operar)."""
    rows = []
    for k, w in base.items():
        if pattern in k and w.is_floating_point():
            d = donor[k].float() - w.float()
            rows.append({"tensor": k, "rel_delta": round((d.norm() / w.float().norm()).item(), 4)})
    return rows


# ---------------- aciertos vs fallos ----------------

def success_failure_contrast(jsonl_path: str, max_examples: int = 3, clip: int = 220) -> dict:
    rows = [json.loads(l) for l in open(jsonl_path)]
    ok = [r for r in rows if r["passed"]]
    bad = [r for r in rows if not r["passed"]]
    err = Counter((r["error"].strip().split("\n")[-1].split(":")[0] if r["error"].strip() else "sin_error") for r in bad)
    L = lambda rs: round(sum(len(r["generation"]) for r in rs) / max(len(rs), 1), 1)
    return {"n": len(rows), "pass": len(ok), "mean_gen_chars_pass": L(ok), "mean_gen_chars_fail": L(bad),
            "fail_error_types": dict(err.most_common(5)),
            "fail_examples": [{"task_id": r["task_id"], "gen": r["generation"][:clip]} for r in bad[:max_examples]]}


# ---------------- historial de intervenciones ----------------

def log_intervention(entry: dict, path: str = "results/metrics/interventions.jsonl") -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def intervention_history(path: str = "results/metrics/interventions.jsonl", last: int = 20) -> list[dict]:
    try:
        return [json.loads(l) for l in open(path)][-last:]
    except FileNotFoundError:
        return []


# ---------------- sensibilidad a parches ----------------

@torch.no_grad()
def corpus_nll(model, tok, dev, texts: list[str], max_len: int = 256) -> float:
    tot, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        lg = model(ids).logits.float()
        tot += torch.nn.functional.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum").item()
        n += ids.shape[1] - 1
    return tot / max(n, 1)


@torch.no_grad()
def patch_sensitivity(model, tok, dev, tensor_name: str, delta: torch.Tensor, scales: list[float],
                      corpora: dict[str, list[str]]) -> dict:
    """NLL por corpus al sumar scale*delta a un tensor (se restaura al final)."""
    param = dict(model.named_parameters())[tensor_name]
    orig = param.data.clone()
    out = {}
    try:
        for s in scales:
            param.data = (orig.float() + s * delta.to(orig.device).float()).to(orig.dtype)
            out[str(s)] = {n: round(corpus_nll(model, tok, dev, t), 5) for n, t in corpora.items()}
    finally:
        param.data = orig
    return out
