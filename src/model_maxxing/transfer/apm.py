"""APM — Activation-Prune-Merge (Fan et al., arXiv 2608.13596), para la familia Qwen2/Qwen2.5.

Transferencia sin entrenamiento de un donante grande (l) a un receptor pequeño (s) de la misma familia:

  1. Perfilado: estadísticas de activación del donante sobre datos de la tarea (Eq. 3, 9-12):
       capa      s = media de ||h_r||_2          (salida de la capa r)
       canal     s = media de |h_r,c|            (ver INTERPRETACIONES)
       cabeza    s = media de ||o_r,k||_2        (salida de la cabeza k antes de o_proj)
       neurona   s = media de |a_r,j|            (ver INTERPRETACIONES)
  2. Poda a la forma del receptor (Eq. 4-6): TopK de capas (en orden del donante), TopK global de canales
     ocultos compartido por todas las capas, TopK de cabezas Q por capa, cabezas KV por la suma de las Q de su
     grupo (GQA), TopK de neuronas MLP por capa (mismos índices en gate/up/down).
  3. Fusión (Eq. 8): θ = (1-μ)·θ_s + μ·θ_slice, μ pequeño (paper: 0.01-0.15).

Línea base IM (Eq. 1-2): mismas reglas de recorte y fusión, pero con las primeras L_s capas y los primeros
índices de cada dimensión (selección fija, independiente de la tarea).

INTERPRETACIONES (el paper no las fija; decisiones nuestras, registradas en el informe):
  - Canal oculto: |h| promediado sobre las salidas de TODAS las capas del donante y sobre los tokens perfilados.
  - Neurona MLP en SwiGLU: a = act(gate(x)) * up(x), es decir, la entrada de down_proj.
  - Cabezas Q y KV se eligen por separado, como en el paper: una cabeza Q puede quedar emparejada con una cabeza
    KV distinta de la de su grupo original. No se corrige.
  - Vocabulario: se toman las primeras V_s filas del embedding del donante (mismos ids de token en Qwen2.5).
  - Receptor con embeddings atados (1.5B): el lm_head del donante no se usa; el embedding fusionado actúa de
    lm_head. Sin reescalado de RMSNorm tras quitar canales (el paper no lo hace).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


# --------------------------------------------------------------------------- arquitectura

@dataclass(frozen=True)
class ArchSpec:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied: bool

    @staticmethod
    def from_config(cfg: dict) -> "ArchSpec":
        if cfg.get("model_type") not in ("qwen2", None):
            raise ValueError(f"APM implementado solo para qwen2, no {cfg.get('model_type')}")
        hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        return ArchSpec(cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"],
                        cfg["num_key_value_heads"], hd, cfg["intermediate_size"], cfg["vocab_size"],
                        bool(cfg.get("tie_word_embeddings", False)))

    @staticmethod
    def from_path(path: str | Path) -> "ArchSpec":
        return ArchSpec.from_config(json.loads((Path(path) / "config.json").read_text()))

    @property
    def group(self) -> int:
        return self.heads // self.kv_heads


def check_compat(donor: ArchSpec, recip: ArchSpec) -> list[str]:
    """Errores que impiden APM (lista vacía = compatible)."""
    err = []
    if donor.head_dim != recip.head_dim:
        err.append(f"head_dim distinto ({donor.head_dim} vs {recip.head_dim}): APM no recorta dentro de una cabeza")
    for f in ("layers", "hidden", "heads", "kv_heads", "intermediate", "vocab"):
        if getattr(donor, f) < getattr(recip, f):
            err.append(f"el donante tiene menos {f} que el receptor ({getattr(donor, f)} < {getattr(recip, f)})")
    return err


# --------------------------------------------------------------------------- estadísticas

@dataclass
class ActivationStats:
    """Sumas acumuladas en float64 (CPU); las medias se obtienen dividiendo por `tokens`."""
    layer: torch.Tensor     # [L]
    hidden: torch.Tensor    # [D]   suma sobre capas y tokens
    head: torch.Tensor      # [L, H]
    mlp: torch.Tensor       # [L, I]
    tokens: int = 0

    @staticmethod
    def zeros(spec: ArchSpec) -> "ActivationStats":
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)
        return ActivationStats(z(spec.layers), z(spec.hidden), z(spec.layers, spec.heads),
                               z(spec.layers, spec.intermediate))

    @torch.no_grad()
    def update(self, r: int, h: torch.Tensor, attn_in: torch.Tensor, mlp_in: torch.Tensor, heads: int) -> None:
        """h [T,D] salida de la capa r; attn_in [T,H*hd] entrada de o_proj; mlp_in [T,I] entrada de down_proj.
        Solo deben llegar los tokens perfilados (Ω(x))."""
        h, attn_in, mlp_in = h.float(), attn_in.float(), mlp_in.float()
        T = h.shape[0]
        acc = lambda x: x.cpu().double()                    # MPS no admite float64: acumular en CPU
        self.layer[r] += acc(h.norm(dim=-1).sum())
        self.hidden += acc(h.abs().sum(0))
        self.head[r] += acc(attn_in.view(T, heads, -1).norm(dim=-1).sum(0))
        self.mlp[r] += acc(mlp_in.abs().sum(0))

    def means(self) -> dict[str, torch.Tensor]:
        z = max(self.tokens, 1)
        L = self.layer.shape[0]
        return {"layer": self.layer / z, "hidden": self.hidden / (z * L), "head": self.head / z, "mlp": self.mlp / z}

    def save(self, path: str | Path) -> str:
        torch.save({**{k: getattr(self, k) for k in ("layer", "hidden", "head", "mlp")}, "tokens": self.tokens},
                   str(path))
        return sha256_file(path)

    @staticmethod
    def load(path: str | Path) -> "ActivationStats":
        d = torch.load(str(path))
        return ActivationStats(d["layer"], d["hidden"], d["head"], d["mlp"], int(d["tokens"]))


# --------------------------------------------------------------------------- selección

@dataclass
class Selection:
    method: str
    layers: list[int]                 # capas del donante, en orden; la i-ésima va a la capa i del receptor
    hidden: list[int]                 # canales ocultos (globales)
    q_heads: list[list[int]]          # por capa del receptor
    kv_heads: list[list[int]]
    mlp: list[list[int]]

    def to_json(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]


def topk_sorted(scores: torch.Tensor, k: int) -> list[int]:
    """TopK determinista (desempate por índice menor) y devuelto en orden creciente de índice."""
    order = torch.argsort(-scores.double(), stable=True)[:k]
    return sorted(order.tolist())


def select_im(donor: ArchSpec, recip: ArchSpec) -> Selection:
    """IM: primeras L_s capas y primeros índices de cada dimensión."""
    r = range
    return Selection("IM", list(r(recip.layers)), list(r(recip.hidden)),
                     [list(r(recip.heads))] * recip.layers, [list(r(recip.kv_heads))] * recip.layers,
                     [list(r(recip.intermediate))] * recip.layers)


def select_apm(stats: ActivationStats, donor: ArchSpec, recip: ArchSpec) -> Selection:
    m = stats.means()
    layers = list(range(donor.layers)) if donor.layers == recip.layers else topk_sorted(m["layer"], recip.layers)
    hidden = topk_sorted(m["hidden"], recip.hidden)
    q, kv, mlp = [], [], []
    for r in layers:
        hs = m["head"][r]
        q.append(topk_sorted(hs, recip.heads))
        kv.append(topk_sorted(hs.view(donor.kv_heads, donor.group).sum(-1), recip.kv_heads))   # S_KV,k = Σ_{h∈G(k)}
        mlp.append(topk_sorted(m["mlp"][r], recip.intermediate))
    return Selection("APM", layers, hidden, q, kv, mlp)


def random_selection(donor: ArchSpec, recip: ArchSpec, seed: int) -> Selection:
    """Control opcional: misma forma, índices aleatorios (no forma parte del paper)."""
    g = torch.Generator().manual_seed(seed)
    pick = lambda n, k: sorted(torch.randperm(n, generator=g)[:k].tolist())
    layers = sorted(pick(donor.layers, recip.layers))
    return Selection(f"RAND{seed}", layers, pick(donor.hidden, recip.hidden),
                     [pick(donor.heads, recip.heads) for _ in layers], [pick(donor.kv_heads, recip.kv_heads) for _ in layers],
                     [pick(donor.intermediate, recip.intermediate) for _ in layers])


# --------------------------------------------------------------------------- recorte

def _head_rows(heads: list[int], hd: int) -> torch.Tensor:
    return torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in heads]) if heads else torch.empty(0, dtype=torch.long)


def recipient_tensor_plan(recip: ArchSpec) -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight"] + ([] if recip.tied else ["lm_head.weight"])
    for i in range(recip.layers):
        p = f"model.layers.{i}."
        names += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
        names += [p + f"self_attn.{x}_proj.{w}" for x in "qkv" for w in ("weight", "bias")]
        names += [p + "self_attn.o_proj.weight"] + [p + f"mlp.{x}_proj.weight" for x in ("gate", "up", "down")]
    return names


def slice_one(name: str, get, sel: Selection, donor: ArchSpec, recip: ArchSpec) -> torch.Tensor:
    """Tensor del receptor `name` construido desde el donante (`get(nombre_donante)`), sin optimización."""
    H = torch.tensor(sel.hidden)
    if name == "model.embed_tokens.weight":
        return get(name)[:recip.vocab][:, H]
    if name == "lm_head.weight":
        src = "model.embed_tokens.weight" if donor.tied else "lm_head.weight"
        return get(src)[:recip.vocab][:, H]
    if name == "model.norm.weight":
        return get(name)[H]
    parts = name.split(".")
    i = int(parts[2])
    src = ".".join(parts[:2] + [str(sel.layers[i])] + parts[3:])
    rest = ".".join(parts[3:])
    w = get(src)
    hd = donor.head_dim
    if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return w[H]
    if rest.startswith("self_attn.q_proj"):
        rows = _head_rows(sel.q_heads[i], hd)
        return w[rows][:, H] if rest.endswith("weight") else w[rows]
    if rest.startswith(("self_attn.k_proj", "self_attn.v_proj")):
        rows = _head_rows(sel.kv_heads[i], hd)
        return w[rows][:, H] if rest.endswith("weight") else w[rows]
    if rest == "self_attn.o_proj.weight":
        return w[H][:, _head_rows(sel.q_heads[i], hd)]
    if rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
        return w[torch.tensor(sel.mlp[i])][:, H]
    if rest == "mlp.down_proj.weight":
        return w[H][:, torch.tensor(sel.mlp[i])]
    raise KeyError(f"tensor no soportado por APM: {name}")


def expected_shape(name: str, recip: ArchSpec) -> tuple[int, ...]:
    D, hd = recip.hidden, recip.head_dim
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (recip.vocab, D)
    if name.endswith("norm.weight"):
        return (D,)
    rest = ".".join(name.split(".")[3:])
    table = {"self_attn.q_proj.weight": (recip.heads * hd, D), "self_attn.q_proj.bias": (recip.heads * hd,),
             "self_attn.k_proj.weight": (recip.kv_heads * hd, D), "self_attn.k_proj.bias": (recip.kv_heads * hd,),
             "self_attn.v_proj.weight": (recip.kv_heads * hd, D), "self_attn.v_proj.bias": (recip.kv_heads * hd,),
             "self_attn.o_proj.weight": (D, recip.heads * hd), "mlp.gate_proj.weight": (recip.intermediate, D),
             "mlp.up_proj.weight": (recip.intermediate, D), "mlp.down_proj.weight": (D, recip.intermediate)}
    return table[rest]


def validate_selection(sel: Selection, donor: ArchSpec, recip: ArchSpec) -> None:
    def ok(idx, n, k, what):
        if len(idx) != k or len(set(idx)) != k or min(idx, default=0) < 0 or max(idx, default=0) >= n:
            raise ValueError(f"selección inválida en {what}: {len(idx)} índices, se esperaban {k} distintos en [0,{n})")
    ok(sel.layers, donor.layers, recip.layers, "capas")
    if sel.layers != sorted(sel.layers):
        raise ValueError("las capas deben conservar el orden del donante")
    ok(sel.hidden, donor.hidden, recip.hidden, "canales")
    for i in range(recip.layers):
        ok(sel.q_heads[i], donor.heads, recip.heads, f"cabezas Q capa {i}")
        ok(sel.kv_heads[i], donor.kv_heads, recip.kv_heads, f"cabezas KV capa {i}")
        ok(sel.mlp[i], donor.intermediate, recip.intermediate, f"MLP capa {i}")


def build_slice(donor_path: str | Path, recip_config: dict, sel: Selection, out_dir: str | Path) -> dict:
    """Escribe el slice del donante con la forma del receptor (dtype del donante, recorte exacto) en un
    directorio NUEVO, con config.json del receptor, la selección y un manifiesto con hashes. No toca originales."""
    from safetensors.torch import save_file

    from model_maxxing.models.loading import CheckpointReader

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} no está vacío: no se sobrescriben slices")
    out.mkdir(parents=True, exist_ok=True)
    donor, recip = ArchSpec.from_path(donor_path), ArchSpec.from_config(recip_config)
    errs = check_compat(donor, recip)
    if errs:
        raise ValueError("; ".join(errs))
    validate_selection(sel, donor, recip)
    tensors = {}
    with CheckpointReader(donor_path) as R:
        for name in recipient_tensor_plan(recip):
            t = slice_one(name, R.get, sel, donor, recip).contiguous().clone()
            if tuple(t.shape) != expected_shape(name, recip):
                raise AssertionError(f"{name}: forma {tuple(t.shape)} != {expected_shape(name, recip)}")
            if not torch.isfinite(t.float()).all():
                raise ValueError(f"NaN/Inf en {name}")
            tensors[name] = t
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(recip_config, indent=1))
    (out / "selection.json").write_text(json.dumps(sel.to_json()))
    manifest = {"method": sel.method, "donor": str(donor_path), "selection_digest": sel.digest(),
                "n_params": sum(t.numel() for t in tensors.values()),
                "dtype": str(next(iter(tensors.values())).dtype),
                "sha256": {f: sha256_file(out / f) for f in ("model.safetensors", "config.json", "selection.json")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def apm_recipe(receiver: str, slice_dir: str, mu: float):
    """Fusión de APM/IM (Eq. 2/8) = transferencia `interpolate` existente con el slice como donante."""
    from model_maxxing.models.loading import Recipe
    if not 0.0 <= mu <= 1.0:
        raise ValueError("μ debe estar en [0,1]")
    return Recipe(receiver=receiver, transfer="interpolate" if mu > 0 else "identity", alpha=mu,
                  donor=slice_dir if mu > 0 else None)


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
