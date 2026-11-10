"""Particiones de datos de v0.4 (pre-registradas). Ver docs/MODEL_SELECTION.md.

  diagnostic  MBPP full train (601-974) + prompt (1-10)        visible para Claude
  selection   MBPP full validation (511-600) + MBPP full test  solo resultados agregados para Claude
              que NO está en sanitized test
  final       MBPP sanitized test[100:] (227-479) + HumanEval+  SELLADO: nunca visible para el agente
  burned      MBPP sanitized test[:100] (v0.2/v0.3)             prohibido en v0.4

Todos los ejemplos MBPP se normalizan a {task_id, prompt, code, test_list, test_imports, source}."""
from __future__ import annotations

import hashlib
import json

MBPP = "google-research-datasets/mbpp"
SPLITS = ("diagnostic", "selection", "final_mbpp", "final_humaneval")
# HumanEval/32: la solución canónica falla con el test de la revisión fijada (TypeError en check) → no resoluble.
HUMANEVAL_EXCLUDED = {"HumanEval/32"}


def _full(split: str) -> list[dict]:
    from datasets import load_dataset
    return [{"task_id": int(r["task_id"]), "prompt": r["text"], "code": r["code"], "test_list": list(r["test_list"]),
             "test_imports": [l for l in (r["test_setup_code"] or "").splitlines() if l.strip()],
             "source": f"mbpp_full/{split}"} for r in load_dataset(MBPP, "full", split=split)]


def _sanitized_test() -> list[dict]:
    from datasets import load_dataset
    return [{"task_id": int(r["task_id"]), "prompt": r["prompt"], "code": r["code"], "test_list": list(r["test_list"]),
             "test_imports": list(r["test_imports"]), "source": "mbpp_sanitized/test"}
            for r in load_dataset(MBPP, "sanitized", split="test")]


def burned_ids() -> set[int]:
    return {r["task_id"] for r in _sanitized_test()[:100]}


def load_split(name: str) -> list[dict]:
    if name == "diagnostic":
        return _full("prompt") + _full("train")
    if name == "selection":
        sanitized = {r["task_id"] for r in _sanitized_test()}
        return _full("validation") + [r for r in _full("test") if r["task_id"] not in sanitized]
    if name == "final_mbpp":
        return _sanitized_test()[100:]
    if name == "final_humaneval":
        from model_maxxing.evaluation.humaneval import load_humanevalplus
        return [r for r in load_humanevalplus() if r["task_id"] not in HUMANEVAL_EXCLUDED]
    raise ValueError(f"partición desconocida: {name}")


def ids_hash(rows: list[dict]) -> str:
    ids = sorted(str(r["task_id"]) for r in rows)
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def check_disjoint(parts: dict[str, list[dict]]) -> None:
    """Las particiones MBPP no comparten task_id entre sí ni con el conjunto quemado."""
    sets = {k: {r["task_id"] for r in v} for k, v in parts.items() if k != "final_humaneval"}
    sets["burned"] = burned_ids()
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                raise ValueError(f"particiones {a} y {b} comparten {len(inter)} ids, p. ej. {sorted(inter)[:5]}")
