"""HumanEval+ (evalplus/humanevalplus, revisión fijada) sobre el mismo sandbox que MBPP.

Formato de completado para modelos base: el prompt es la firma + docstring y el modelo continúa el cuerpo.
El programa ejecutado es  prompt + completado + test (define check) + check(entry_point).
Así se reutiliza `runner.run_tests` sin cambios (el test va como una "línea" más de test_list)."""
from __future__ import annotations

REPO = "evalplus/humanevalplus"
REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
# Paradas habituales para completado de HumanEval con modelos base (columna 0 = fin de la función).
STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n```", "\nassert "]
TIMEOUT = 20          # las entradas "plus" son muchas: más margen que MBPP (10 s)


def load_humanevalplus() -> list[dict]:
    from datasets import load_dataset
    return [{"task_id": r["task_id"], "prompt": r["prompt"], "code": r["canonical_solution"],
             "entry_point": r["entry_point"], "test": r["test"], "source": "humanevalplus"}
            for r in load_dataset(REPO, split="test", revision=REVISION)]


def build_prompt(ex: dict) -> str:
    return ex["prompt"]


def extract_code(text: str) -> str:
    cut = min([i for i in (text.find(s) for s in STOPS) if i >= 0], default=len(text))
    return text[:cut]


def exec_spec(ex: dict, completion: str) -> tuple[str, dict]:
    """(código, ejemplo-para-run_tests) equivalente a la evaluación de HumanEval+."""
    code = ex["prompt"] + completion
    return code, {"test_imports": [], "test_list": [ex["test"], f"check({ex['entry_point']})"]}


def run(ex: dict, completion: str) -> tuple[bool, str]:
    from model_maxxing.evaluation.runner import run_tests
    code, spec = exec_spec(ex, completion)
    return run_tests(code, spec, timeout=TIMEOUT)
