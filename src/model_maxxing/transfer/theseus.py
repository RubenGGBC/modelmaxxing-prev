"""Theseus (arXiv 2602.12952): traslado de task vectors entre anchuras distintas sin entrenamiento, para Qwen2.

  τ_A = θ_A^ajustado − θ_A^base                          (p. ej. Qwen2.5-Coder-7B − Qwen2.5-7B)
  Por capa lineal ℓ y con los MISMOS tokens en A (base) y B (receptor) (ver OUT_SPACE para gate/up):
     C_in = Σ H_in,Aᵀ H_in,B ,   C_out = Σ H_out,Aᵀ H_out,B
     T = U Vᵀ  con  C = U Σ Vᵀ  (Procrustes ortogonal: argmin ||H_A T − H_B||, TᵀT = I)
  τ_B = T_outᵀ · τ_A · T_in      (la Eq. 10 del paper tiene las transposiciones inconsistentes; esta es la
                                  forma coherente: si x_B ≈ x_A T_in, entonces Δy_B ≈ x_B T_inᵀ τ_Aᵀ T_out)
  Receptor final: θ_B + α·τ_B.

Decisiones nuestras donde el paper no especifica (ver docs/HITO_B_THESEUS_PLAN.md):
  - Se trasladan q,k,v,o,gate,up,down y los sesgos q/k/v (τ_b,B = τ_b,A · T_out). NO embeddings, lm_head ni normas.
  - q/k/v comparten entrada (una T_in "attn_in"); gate/up también ("mlp_in").
  - Salida de gate/up = espacio de neuronas del MLP, alineado con la entrada de down_proj (ver OUT_SPACE).
  - Con anchuras distintas, T = UVᵀ maximiza tr(TᵀC) (correlación). Solo coincide con el mínimo de
    ||H_A T − H_B|| si H_Aᵀ H_A ∝ I (activaciones isotrópicas); las de Qwen no lo son. Es el método del paper.
  - Capas emparejadas por índice (exige igual profundidad). Mismo tokenizer → tokens emparejados 1:1.
  - La variante sin q/k (RoPE) se obtiene excluyendo esas claves al aplicar el delta.

Controles: NAIVE (τ_A truncado a la forma de B, sin alinear) y RAND (ruido con la norma de τ_A por tensor,
trasladado por las MISMAS T)."""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from model_maxxing.models.loading import CheckpointReader
from model_maxxing.transfer.apm import ArchSpec, sha256_file

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODULE = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
          "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
IN_GROUP = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
            "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in"}
# Espacio de SALIDA de cada Linear. gate/up expanden (d_out > d_in): su salida tiene rango ≤ d_in y alinearla por
# separado no está identificado (Procrustes elegiría al azar la mayor parte de la rotación). gate.out, up.out y
# down.in indexan el MISMO espacio de neuronas: se alinea una vez con la post-activación (entrada de down_proj),
# de rango completo por la no linealidad. Desviación documentada respecto a "entrada y salida de cada Linear".
OUT_SPACE = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "o_proj": "o_proj",
             "gate_proj": "down_in", "up_proj": "down_in", "down_proj": "down_proj"}
BIASED = ("q_proj", "k_proj", "v_proj")
QK_KEYS = ("self_attn.q_proj.", "self_attn.k_proj.")


def tensor_name(r: int, lin: str, kind: str = "weight") -> str:
    return f"model.layers.{r}.{MODULE[lin]}.{lin}.{kind}"


# --------------------------------------------------------------------------- álgebra (pura, testeable)

def procrustes_sv(C: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """(T = U Vᵀ, Σσ, nº de direcciones no identificables) de una SVD fina de C (d_A × d_B), SIEMPRE en float64.

    Las covarianzas de Qwen tienen condicionamiento 1e9-1e12 (sesgos enormes en q/k, activaciones masivas):
    en float32 la rotación sale arbitraria en la mayoría de direcciones (medido en 0.5B, docs del Hito B).
    Direcciones no identificables = rango numérico deficiente (σ < σmax·max(m,n)·ε_f64, criterio de matrix_rank):
    sin energía en los datos, ahí T no está determinada."""
    Cc = C.double() if C.device.type == "cuda" else C.cpu().double()     # MPS no tiene float64 ni SVD
    U, S, Vh = torch.linalg.svd(Cc, full_matrices=False)
    tol = S[0] * max(Cc.shape) * torch.finfo(torch.float64).eps if S.numel() else 0.0
    null = int((S <= tol).sum().item()) if S.numel() else 0
    return (U @ Vh).float(), S.sum().item(), null


def procrustes(C: torch.Tensor) -> torch.Tensor:
    """T = U Vᵀ de la SVD fina de C. Columnas (o filas) ortonormales."""
    return procrustes_sv(C)[0].to(C.device)


def procrustes_similarity(C: torch.Tensor, ssq_a: float, ssq_b: float) -> float:
    """Σσ(C) / sqrt(||H_A||² ||H_B||²) ∈ [0,1]: 1 = B es exactamente una rotación de A (diagnóstico)."""
    s = torch.linalg.svdvals(C.double() if C.device.type == "cuda" else C.cpu().double()).sum().item()
    return s / max((ssq_a * ssq_b) ** 0.5, 1e-30)


def transport(tau: torch.Tensor, T_in: torch.Tensor, T_out: torch.Tensor) -> torch.Tensor:
    return T_out.T @ tau @ T_in


def transport_bias(tau_b: torch.Tensor, T_out: torch.Tensor) -> torch.Tensor:
    return tau_b @ T_out


def naive(tau: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return tau[:shape[0]] if len(shape) == 1 else tau[:shape[0], :shape[1]]


def matched_noise(tau: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    """Ruido gaussiano determinista (por nombre de tensor y semilla) con la misma norma de Frobenius que τ."""
    h = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:15], 16)
    g = torch.randn(tau.shape, generator=torch.Generator().manual_seed(h), dtype=torch.float32)
    return g * (tau.float().norm() / g.norm().clamp_min(1e-30)).cpu()


# --------------------------------------------------------------------------- alineamiento en streaming

def _causal_mask(T: int, dtype, device):
    return torch.full((T, T), torch.finfo(dtype).min, dtype=dtype, device=device).triu(1)[None, None]


class _Model:
    """Un modelo recorrido capa a capa (como apm_profile), con captura de entradas/salidas de los Linear."""

    def __init__(self, path, examples, device, dtype):
        from transformers import AutoConfig
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        self.path, self.device, self.dtype = Path(path), device, dtype
        self.cfg = AutoConfig.from_pretrained(str(path))
        self.cfg._attn_implementation = "eager"
        self.spec = ArchSpec.from_path(path)
        self.reader = CheckpointReader(path)
        emb = self.reader.get("model.embed_tokens.weight")
        self.hs = [emb[torch.tensor(ids)].to(device=device, dtype=dtype)[None] for ids in examples]
        del emb
        rot = Qwen2RotaryEmbedding(self.cfg).to(device)
        self.aux = []
        for h in self.hs:
            T = h.shape[1]
            pos = torch.arange(T, device=device)[None]
            self.aux.append((_causal_mask(T, dtype, device), pos, rot(h, pos)))

    def layer(self, r):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        layer = Qwen2DecoderLayer(self.cfg, r)
        pre = f"model.layers.{r}."
        layer.load_state_dict({k[len(pre):]: self.reader.get(k) for k in self.reader.keys() if k.startswith(pre)},
                              strict=True)
        return layer.to(device=self.device, dtype=self.dtype).eval()

    def run(self, layer, j):
        """Pasa el ejemplo j por la capa; devuelve {lin: (entrada [T,d_in], salida [T,d_out])}."""
        cap, hooks = {}, []
        for lin in LINEARS:
            m = getattr(getattr(layer, MODULE[lin]), lin)
            hooks.append(m.register_forward_hook(lambda mod, a, o, lin=lin: cap.__setitem__(lin, (a[0][0], o[0]))))
        mask, pos, pe = self.aux[j]
        out = layer(self.hs[j], attention_mask=mask, position_ids=pos, position_embeddings=pe)
        self.hs[j] = out[0] if isinstance(out, tuple) else out
        for h in hooks:
            h.remove()
        return cap

    def close(self):
        self.reader.__exit__()


@torch.no_grad()
def align_and_transport(base_a: str, ft_a: str, recv_b: str, examples: list[list[int]], out_root: str | Path,
                        device: str, dtype=torch.float32, rand_seed: int = 0, log=print) -> dict:
    """Recorre A (base) y B capa a capa con los mismos tokens, calcula las T por Procrustes y escribe tres deltas
    con la forma de B en `out_root/{TH,RAND,NAIVE}` (un safetensors por capa + índice). No toca originales."""
    from safetensors.torch import save_file

    out_root = Path(out_root)
    arms = ("TH", "RAND", "NAIVE")
    for a in arms:
        d = out_root / a
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"{d} no está vacío: no se sobrescriben deltas")
        d.mkdir(parents=True, exist_ok=True)
    sa, sb, sf = ArchSpec.from_path(base_a), ArchSpec.from_path(recv_b), ArchSpec.from_path(ft_a)
    if sa != sf:
        raise ValueError(f"base y ajustado de A con arquitecturas distintas: {sa} vs {sf}")
    if sa.layers != sb.layers:
        raise ValueError(f"Theseus exige igual profundidad ({sa.layers} vs {sb.layers})")
    A, B = _Model(base_a, examples, device, dtype), _Model(recv_b, examples, device, dtype)
    weight_maps = {a: {} for a in arms}
    diag = {"layers": [], "tokens": sum(len(e) for e in examples), "n_examples": len(examples)}
    t_all = time.time()
    with CheckpointReader(ft_a) as FT:
        for r in range(sa.layers):
            t0 = time.time()
            la, lb = A.layer(r), B.layer(r)
            C, ssq = {}, {}
            acc = "cpu" if device == "mps" else device          # float64: MPS no lo soporta
            for j in range(len(examples)):
                ca, cb = A.run(la, j), B.run(lb, j)
                pairs = {IN_GROUP[l]: (ca[l][0], cb[l][0]) for l in LINEARS}            # entradas por grupo
                pairs.update({l: (ca[l][1], cb[l][1]) for l in LINEARS if OUT_SPACE[l] == l})   # salidas propias
                for g, (xa, xb) in pairs.items():
                    xa, xb = xa.to(acc).double(), xb.to(acc).double()        # mover y luego convertir (MPS)
                    if g not in C:
                        C[g] = torch.zeros(xa.shape[1], xb.shape[1], device=acc, dtype=torch.float64)
                        ssq[g] = [0.0, 0.0]
                    C[g] += xa.T @ xb
                    ssq[g][0] += xa.pow(2).sum().item()
                    ssq[g][1] += xb.pow(2).sum().item()
            del la, lb
            T, sim, null = {}, {}, {}
            for g, c in C.items():
                T[g], ssum, null[g] = procrustes_sv(c)
                T[g] = T[g].to(device)
                sim[g] = round(ssum / max((ssq[g][0] * ssq[g][1]) ** 0.5, 1e-30), 4)   # similitud ∈ [0,1]
            del C
            out = {a: {} for a in arms}
            norms = {}
            for lin in LINEARS:
                for kind in ("weight", "bias") if lin in BIASED else ("weight",):
                    name = tensor_name(r, lin, kind)
                    tau = (FT.get(name).float() - A.reader.get(name).float())
                    shape = tuple(B.reader.get(name).shape)
                    noise = matched_noise(tau, name, rand_seed)
                    tau, noise = tau.to(device), noise.to(device)
                    T_out, T_in = T[OUT_SPACE[lin]], T[IN_GROUP[lin]]
                    if kind == "weight":
                        th, rd = transport(tau, T_in, T_out), transport(noise, T_in, T_out)
                    else:
                        th, rd = transport_bias(tau, T_out), transport_bias(noise, T_out)
                    nv = naive(tau, shape)
                    for a, t in (("TH", th), ("RAND", rd), ("NAIVE", nv)):
                        if tuple(t.shape) != shape or not torch.isfinite(t).all():
                            raise AssertionError(f"{a} {name}: forma {tuple(t.shape)} != {shape} o no finito")
                        out[a][name] = t.float().cpu().contiguous()
                    norms[name] = {"tau_A": round(tau.norm().item(), 5), "TH": round(th.norm().item(), 5),
                                   "NAIVE": round(nv.norm().item(), 5), "B": round(B.reader.get(name).float().norm().item(), 5)}
            for a in arms:
                fn = f"layer_{r:02d}.safetensors"
                save_file(out[a], str(out_root / a / fn), metadata={"format": "pt"})
                weight_maps[a].update({k: fn for k in out[a]})
            del T, out
            if device == "mps":
                torch.mps.empty_cache()
            elif device.startswith("cuda"):
                torch.cuda.empty_cache()
            diag["layers"].append({"layer": r, "seconds": round(time.time() - t0, 2), "procrustes_similarity": sim,
                                   "null_dims": null, "norms": norms})
            log(f"  capa {r + 1}/{sa.layers}: {time.time() - t0:.1f} s  sim mlp_in={sim['mlp_in']} down_in={sim['down_in']}")
    A.close()
    B.close()
    diag["total_seconds"] = round(time.time() - t_all, 1)
    manifests = {}
    for a in arms:
        d = out_root / a
        (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_maps[a]}))
        manifests[a] = {"arm": a, "base_a": str(base_a), "ft_a": str(ft_a), "recv_b": str(recv_b),
                        "rand_seed": rand_seed if a == "RAND" else None, "n_tensors": len(weight_maps[a]),
                        "sha256": {p.name: sha256_file(p) for p in sorted(d.glob("*.safetensors"))}}
        (d / "manifest.json").write_text(json.dumps(manifests[a], indent=1))
    (out_root / "alignment_diagnostics.json").write_text(json.dumps(diag, indent=1))
    return {"manifests": manifests, "total_seconds": diag["total_seconds"]}


# --------------------------------------------------------------------------- aplicar al receptor

@torch.no_grad()
def materialize_affine(model, receiver: str, delta_dir: str | None = None, scale: float = 1.0, alpha: float = 0.0,
                       exclude: tuple[str, ...] = ()) -> dict:
    """W_k = scale·R_k + alpha·D_k  (D_k solo si k está en el delta y no excluido), en FP32.
    Cubre todas las variantes:  Theseus (scale=1, alpha=α), APM/IM (scale=1−μ, alpha=μ, D=slice),
    solo-encoger (scale=1−μ, sin D) y receptor original (scale=1, sin D). No toca los ficheros originales."""
    params = dict(model.named_parameters())
    applied = 0.0
    used = 0
    with ExitStack() as st:
        R = st.enter_context(CheckpointReader(receiver))
        D = st.enter_context(CheckpointReader(delta_dir)) if delta_dir else None
        dkeys = set(D.keys()) if D else set()
        unknown = {k for k in dkeys if k not in params}
        if unknown:
            raise KeyError(f"claves del delta que el receptor no tiene: {sorted(unknown)[:5]}")
        for k, p in params.items():
            w = R.get(k).float() * scale
            if k in dkeys and not any(x in k for x in exclude):
                d = D.get(k).float()
                w = w + alpha * d
                applied += (alpha * d).pow(2).sum().item()
                used += 1
            if not torch.isfinite(w).all():
                raise ValueError(f"NaN/Inf en {k}")
            p.data.copy_(w.to(p.device, p.dtype))
    return {"delta_norm_applied": applied ** 0.5, "delta_tensors_applied": used}
