"""Perfilado APM del donante capa a capa (streaming desde safetensors).

Nunca hay más de UNA capa del donante en memoria: los estados ocultos de todos los ejemplos se mantienen en el
dispositivo y se pasan por la capa r, que luego se libera. Memoria pico ≈ embedding + 1 capa + estados ocultos.
Para Qwen2.5-7B en bf16: 1,09 GB (embedding, se libera tras usarlo) + 0,47 GB/capa + N·T·3584·4 B.

Usa las clases de capa de transformers (atención eager, máscara causal explícita, batch 1 sin relleno), por lo
que es numéricamente equivalente al forward de HF; tests/test_apm.py lo comprueba contra el modelo completo."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from model_maxxing.models.loading import CheckpointReader
from model_maxxing.transfer.apm import ActivationStats, ArchSpec


def _causal_mask(T: int, dtype, device) -> torch.Tensor:
    m = torch.full((T, T), torch.finfo(dtype).min, dtype=dtype, device=device).triu(1)
    return m[None, None]


def _layer_config(path: Path):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(path))
    cfg._attn_implementation = "eager"
    return cfg


@torch.no_grad()
def profile_donor(donor_path: str | Path, examples: list[tuple[list[int], list[bool]]], device: str,
                  dtype=torch.bfloat16, log=print) -> tuple[ActivationStats, dict]:
    """examples: [(input_ids, profile_mask)]; profile_mask marca las posiciones Ω(x) que se perfilan.
    Devuelve las estadísticas y un registro de coste (tiempo por capa, memoria)."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

    path = Path(donor_path)
    spec = ArchSpec.from_path(path)
    cfg = _layer_config(path)
    stats = ActivationStats.zeros(spec)
    stats.tokens = int(sum(sum(m) for _, m in examples))
    rotary = Qwen2RotaryEmbedding(cfg).to(device)
    cost = {"layer_seconds": [], "n_examples": len(examples), "tokens_profiled": stats.tokens,
            "tokens_total": sum(len(i) for i, _ in examples), "dtype": str(dtype), "device": device}
    t_start = time.time()
    with CheckpointReader(path) as R:
        emb = R.get("model.embed_tokens.weight")
        hs = [emb[torch.tensor(ids)].to(device=device, dtype=dtype)[None] for ids, _ in examples]
        del emb
        masks = [torch.tensor(m, device=device) for _, m in examples]
        aux = []
        for h in hs:
            T = h.shape[1]
            pos = torch.arange(T, device=device)[None]
            aux.append((_causal_mask(T, dtype, device), pos, rotary(h, pos)))
        for r in range(spec.layers):
            t0 = time.time()
            layer = Qwen2DecoderLayer(cfg, r)
            prefix = f"model.layers.{r}."
            layer.load_state_dict({k[len(prefix):]: R.get(k) for k in R.keys() if k.startswith(prefix)}, strict=True)
            layer = layer.to(device=device, dtype=dtype).eval()
            cap = {}
            hk = [layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("attn", a[0])),
                  layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("mlp", a[0]))]
            for j, (h, (mask, pos, pe)) in enumerate(zip(hs, aux)):
                out = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
                out = out[0] if isinstance(out, tuple) else out
                sel = masks[j]
                stats.update(r, out[0][sel], cap["attn"][0][sel], cap["mlp"][0][sel], spec.heads)
                hs[j] = out
            for x in hk:
                x.remove()
            del layer
            if device == "mps":
                torch.mps.empty_cache()
            elif device.startswith("cuda"):
                torch.cuda.empty_cache()
            cost["layer_seconds"].append(round(time.time() - t0, 3))
            log(f"  capa {r + 1}/{spec.layers}: {cost['layer_seconds'][-1]} s")
    cost["total_seconds"] = round(time.time() - t_start, 2)
    cost["memory"] = device_memory(device)
    return stats, cost


def device_memory(device: str) -> dict:
    import resource
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes en macOS, KiB en Linux
    out = {"process_max_rss_gb": round(rss / (2**30 if sys.platform == "darwin" else 2**20), 2)}
    if device == "mps":
        out["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
    elif device.startswith("cuda"):
        out["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def write_stats(stats: ActivationStats, cost: dict, out_dir: str | Path, meta: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = stats.save(out / "activation_stats.pt")
    rec = {"stats_sha256": h, "cost": cost, **meta}
    (out / "profile_meta.json").write_text(json.dumps(rec, indent=1))
    return rec
