"""Esquema tipado de parches. Solo ediciones numéricas explícitas: sin código arbitrario."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockEdit:
    """Suma `values` (matriz explícita) a state[tensor][row0:row0+h, col0:col0+w]."""
    tensor: str
    row0: int
    col0: int
    values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class LowRankEdit:
    """Suma scale * u v^T a state[tensor][row0:row0+len(u), col0:col0+len(v)]."""
    tensor: str
    row0: int
    col0: int
    u: tuple[float, ...]
    v: tuple[float, ...]
    scale: float = 1.0


@dataclass(frozen=True)
class PatchSpec:
    edits: tuple[BlockEdit | LowRankEdit, ...]
    parent_id: str = "receiver"
    justification: str = ""
    author: str = "claude"
    max_relative_norm: float = 0.05   # ||P||_F / ||W_tensor||_F por tensor
    allowed_pattern: str = r"model\.layers\.\d+\.(self_attn|mlp)\.\w+\.weight"
    meta: dict = field(default_factory=dict)


def edit_to_dict(e) -> dict:
    d = {k: getattr(e, k) for k in e.__dataclass_fields__}
    d["kind"] = type(e).__name__
    return d


def spec_to_dict(s: PatchSpec) -> dict:
    return {"edits": [edit_to_dict(e) for e in s.edits], "parent_id": s.parent_id,
            "justification": s.justification, "author": s.author,
            "max_relative_norm": s.max_relative_norm, "allowed_pattern": s.allowed_pattern,
            "meta": s.meta}


def spec_from_dict(d: dict) -> PatchSpec:
    edits = []
    for e in d["edits"]:
        e = dict(e)
        kind = e.pop("kind")
        if kind == "BlockEdit":
            e["values"] = tuple(tuple(r) for r in e["values"])
            edits.append(BlockEdit(**e))
        elif kind == "LowRankEdit":
            e["u"], e["v"] = tuple(e["u"]), tuple(e["v"])
            edits.append(LowRankEdit(**e))
        else:
            raise ValueError(f"Tipo de edición no permitido: {kind}")
    return PatchSpec(edits=tuple(edits), **{k: d[k] for k in
                     ("parent_id", "justification", "author", "max_relative_norm", "allowed_pattern", "meta")})
