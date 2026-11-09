"""Inspección de compatibilidad entre checkpoints (sin cargar pesos completos)."""
from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

ARCH_KEYS = [
    "architectures", "model_type", "num_hidden_layers", "hidden_size",
    "num_attention_heads", "num_key_value_heads", "intermediate_size",
    "vocab_size", "tie_word_embeddings", "torch_dtype",
]


class IncompatibleModels(ValueError):
    pass


def _shards(path: Path) -> list[Path]:
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"Sin .safetensors en {path}")
    return files


def tensor_shapes(path: str | Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Forma y dtype de cada tensor leyendo solo cabeceras."""
    out = {}
    for f in _shards(Path(path)):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                s = sf.get_slice(k)
                out[k] = (tuple(s.get_shape()), s.get_dtype())
    return out


def read_config(path: str | Path) -> dict:
    cfg = json.loads((Path(path) / "config.json").read_text())
    return {k: cfg.get(k) for k in ARCH_KEYS}


def tokenizer_fingerprint(path: str | Path) -> str | None:
    import hashlib
    f = Path(path) / "tokenizer.json"
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None


def compare_shapes(a: dict, b: dict) -> dict:
    ka, kb = set(a), set(b)
    mismatched = sorted(k for k in ka & kb if a[k][0] != b[k][0])
    return {
        "only_in_a": sorted(ka - kb),
        "only_in_b": sorted(kb - ka),
        "shape_mismatch": mismatched,
        "compatible": not (ka ^ kb) and not mismatched,
    }


def check_pair(path_a: str | Path, path_b: str | Path) -> dict:
    sa, sb = tensor_shapes(path_a), tensor_shapes(path_b)
    ca, cb = read_config(path_a), read_config(path_b)
    ta, tb = tokenizer_fingerprint(path_a), tokenizer_fingerprint(path_b)
    shapes = compare_shapes(sa, sb)
    cfg_diff = {k: (ca[k], cb[k]) for k in ARCH_KEYS if ca[k] != cb[k] and k != "torch_dtype"}
    return {
        "shapes": shapes,
        "config_diff": cfg_diff,
        "tokenizer_identical": ta is not None and ta == tb,
        "compatible": shapes["compatible"] and not cfg_diff and ta == tb,
    }


def require_compatible(state_a: dict, state_b: dict) -> None:
    """Rechaza operaciones entre state_dicts con claves o formas distintas."""
    if set(state_a) != set(state_b):
        diff = sorted(set(state_a) ^ set(state_b))[:5]
        raise IncompatibleModels(f"Claves distintas, p.ej. {diff}")
    for k in state_a:
        if tuple(state_a[k].shape) != tuple(state_b[k].shape):
            raise IncompatibleModels(f"Forma distinta en {k}: {tuple(state_a[k].shape)} vs {tuple(state_b[k].shape)}")
