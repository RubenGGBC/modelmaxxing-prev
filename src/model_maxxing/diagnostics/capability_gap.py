"""Brecha funcional R vs D por subhabilidad (Hito A / B).

Las etiquetas de subhabilidad son reglas deterministas sobre la solución de referencia y el enunciado
(multi-etiqueta). Son heurísticas: sirven para localizar dónde gana D, no como taxonomía exacta."""
from __future__ import annotations

import ast
import re

SUBSKILLS = ("recursion", "string", "list_transform", "math", "dict_set", "sorting", "regex", "loops_nested",
             "tuple", "bit_ops")


def _defs_and_calls(code: str) -> tuple[set[str], dict[str, set[str]]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), {}
    defs, calls = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defs.add(node.name)
            calls[node.name] = {n.func.id for n in ast.walk(node)
                                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return defs, calls


def _nested_loops(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    loops = (ast.For, ast.While, ast.comprehension)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            if any(isinstance(n, loops) for n in ast.walk(node) if n is not node):
                return True
    return False


def tag_subskills(code: str, text: str = "") -> list[str]:
    t = text.lower()
    defs, calls = _defs_and_calls(code)
    tags = []
    if any(name in calls.get(name, set()) for name in defs):
        tags.append("recursion")
    if re.search(r"\bstr\b|string|\.split\(|\.join\(|\.upper\(|\.lower\(|\.replace\(|\.strip\(", code) or \
            re.search(r"\bstring|\bword|\bcharacter|\bvowel", t):
        tags.append("string")
    if re.search(r"\[[^\]]*\bfor\b[^\]]*\]|\bmap\(|\bfilter\(|\.append\(|\bzip\(", code) or re.search(r"\blist", t):
        tags.append("list_transform")
    if re.search(r"\bmath\.|\*\*|%|\bsqrt\b|\bfactorial\b|\bprime|\bgcd\b", code) or \
            re.search(r"\bnumber|\bsum\b|\bprime|\bdigit|\bdivisor|\bperimeter|\barea\b", t):
        tags.append("math")
    if re.search(r"\bdict\(|\{\s*\}|\bset\(|\.items\(\)|\.keys\(\)|Counter|defaultdict", code) or \
            re.search(r"\bdictionar|\bset\b", t):
        tags.append("dict_set")
    if re.search(r"\bsorted\(|\.sort\(|\bheapq\b", code) or re.search(r"\bsort", t):
        tags.append("sorting")
    if re.search(r"\bimport re\b|\bre\.", code) or re.search(r"\bregex", t):
        tags.append("regex")
    if _nested_loops(code):
        tags.append("loops_nested")
    if re.search(r"\btuple\(", code) or re.search(r"\btuple", t):
        tags.append("tuple")
    if re.search(r"<<|>>|\^|(?<![&|])&(?!&)|(?<![|])\|(?!\|)", code) or re.search(r"\bbit", t):
        tags.append("bit_ops")
    return tags


def gap_by_subskill(examples: list[dict], passed_r: dict, passed_d: dict) -> dict:
    """Por subhabilidad: n, aciertos R y D, R-falla/D-acierta y viceversa. passed_*: {task_id: bool}."""
    out = {}
    for s in SUBSKILLS + ("all",):
        ids = [e["task_id"] for e in examples if s == "all" or s in tag_subskills(e["code"], e["prompt"])]
        ids = [i for i in ids if i in passed_r and i in passed_d]
        if not ids:
            continue
        r = sum(passed_r[i] for i in ids)
        d = sum(passed_d[i] for i in ids)
        out[s] = {"n": len(ids), "R": r, "D": d, "gap_pp": round(100 * (d - r) / len(ids), 1),
                  "R_fail_D_pass": sum(passed_d[i] and not passed_r[i] for i in ids),
                  "R_pass_D_fail": sum(passed_r[i] and not passed_d[i] for i in ids)}
    return out
