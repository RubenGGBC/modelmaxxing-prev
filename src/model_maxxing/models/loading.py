"""Carga y materialización de variantes sin tener B, R y D completos en memoria a la vez (v0.4).

- Los pesos se leen tensor a tensor de los safetensors (mmap) y se combinan en FP32.
- El modelo vive en bf16 en GPU, salvo los módulos operables (`fp32_modules`), que se suben a FP32 en
  TODAS las variantes para que parches pequeños no se pierdan por redondeo y la comparación sea justa.
- Receta: W = Transfer(R + P, D), con Transfer en {identity, task_vector, interpolate}."""
from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open


class CheckpointReader:
    """Acceso perezoso por nombre de tensor, con o sin índice de shards."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        idx = self.path / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            files = sorted(self.path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"Sin .safetensors en {self.path}")
            wm = {}
            for f in files:
                with safe_open(str(f), "pt") as sf:
                    wm.update({k: f.name for k in sf.keys()})
        self.weight_map = wm
        self._stack = ExitStack()
        self._open = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._stack.close()

    def keys(self):
        return self.weight_map.keys()

    def get(self, name: str) -> torch.Tensor:
        fn = self.weight_map[name]
        if fn not in self._open:
            self._open[fn] = self._stack.enter_context(safe_open(str(self.path / fn), "pt"))
        return self._open[fn].get_tensor(name)


@dataclass
class Recipe:
    """Variante reproducible. patch: {tensor: delta FP32}; se suma a R ANTES de transferir."""
    receiver: str
    transfer: str = "identity"            # identity | task_vector | interpolate
    alpha: float = 0.0
    donor: str | None = None
    base: str | None = None
    patch: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {"receiver": self.receiver, "transfer": self.transfer, "alpha": self.alpha, "donor": self.donor,
                "base": self.base, "patched_tensors": sorted(self.patch)}


def combine(r: torch.Tensor, recipe: Recipe, d: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor:
    """Fórmula de transferencia en FP32 (pura, testeable)."""
    if recipe.transfer == "identity":
        return r
    if recipe.transfer == "task_vector":
        return r + recipe.alpha * (d - b)
    if recipe.transfer == "interpolate":
        return (1 - recipe.alpha) * r + recipe.alpha * d
    raise ValueError(f"transferencia desconocida: {recipe.transfer}")


def upcast_modules(model, module_names: list[str]) -> None:
    """Sube a FP32 los pesos de los Linear indicados; castea entrada a FP32 y salida al dtype del modelo."""
    mods = dict(model.named_modules())
    for name in module_names:
        m = mods[name]
        if getattr(m, "_mm_fp32", False):
            continue
        out_dtype = m.weight.dtype
        m.weight.data = m.weight.data.float()
        if m.bias is not None:
            m.bias.data = m.bias.data.float()
        m.register_forward_pre_hook(lambda mod, args: tuple(a.float() if torch.is_tensor(a) else a for a in args))
        m.register_forward_hook(lambda mod, args, out, dt=out_dtype: out.to(dt))
        m._mm_fp32 = True


@torch.no_grad()
def materialize(model, recipe: Recipe) -> dict:
    """Escribe en los parámetros del modelo W = Transfer(R + P, D). Devuelve norma del parche aplicada."""
    params = dict(model.named_parameters())
    unknown = set(recipe.patch) - set(params)
    if unknown:
        raise KeyError(f"tensores del parche inexistentes: {sorted(unknown)}")
    with ExitStack() as st:
        R = st.enter_context(CheckpointReader(recipe.receiver))
        D = st.enter_context(CheckpointReader(recipe.donor)) if recipe.transfer != "identity" else None
        B = st.enter_context(CheckpointReader(recipe.base)) if recipe.transfer == "task_vector" else None
        missing = set(params) - set(R.keys())
        if missing:
            raise KeyError(f"parámetros sin peso en el receptor: {sorted(missing)[:5]}")
        applied = 0.0
        for k, p in params.items():
            r = R.get(k).float()
            if k in recipe.patch:
                r = r + recipe.patch[k].float()
            w = combine(r, recipe, D.get(k).float() if D else None, B.get(k).float() if B else None)
            if not torch.isfinite(w).all():
                raise ValueError(f"NaN/Inf en {k}")
            p.data.copy_(w.to(p.device, p.dtype))
            if k in recipe.patch:
                ref = combine(R.get(k).float(), recipe, D.get(k).float() if D else None, B.get(k).float() if B else None)
                applied += (p.data.float().cpu() - ref.to(p.dtype).float()).pow(2).sum().item()
    return {"applied_patch_norm_after_transfer": applied ** 0.5}


def load_for_variants(path: str, device: str, fp32_modules: list[str], dtype=torch.bfloat16):
    """Esqueleto del modelo (pesos de `path`) en `dtype`, con los módulos operables en FP32."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    upcast_modules(model, fp32_modules)
    return model, tok


def dtype_report(model) -> dict:
    out = {}
    for k, p in model.named_parameters():
        out.setdefault(str(p.dtype), 0)
        out[str(p.dtype)] += p.numel()
    return out


def memory_report() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    free, total = torch.cuda.mem_get_info()
    return {"device": torch.cuda.get_device_name(0), "vram_total_gb": round(total / 2**30, 2),
            "vram_free_gb": round(free / 2**30, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def require_vram(gb_needed: float) -> None:
    """Falla de forma controlada si no hay VRAM libre suficiente (en vez de un OOM a mitad de ejecución)."""
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free < gb_needed:
            raise MemoryError(f"VRAM libre {free:.2f} GB < {gb_needed:.2f} GB necesarios")
