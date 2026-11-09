"""MedMCQA 3-shot por log-probabilidad de la letra. Resultados por ítem."""
from __future__ import annotations

import json

import torch
from datasets import load_dataset

LETTERS = ["A", "B", "C", "D"]


def load_medmcqa(split: str, start: int, n: int) -> list[dict]:
    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    return [ds[i] for i in range(start, start + n)]


def fmt(ex: dict, with_answer: bool) -> str:
    q = f"Question: {ex['question']}\nA. {ex['opa']}\nB. {ex['opb']}\nC. {ex['opc']}\nD. {ex['opd']}\nAnswer:"
    return q + (f" {LETTERS[ex['cop']]}\n\n" if with_answer else "")


def fewshot_prefix(shots: list[dict]) -> str:
    return "The following are multiple choice questions about medicine.\n\n" + "".join(fmt(s, True) for s in shots)


@torch.no_grad()
def eval_medmcqa(model, tok, dev, items: list[dict], shots: list[dict], out_jsonl: str | None = None) -> list[dict]:
    prefix = fewshot_prefix(shots)
    letter_ids = [tok(" " + l).input_ids[-1] for l in LETTERS]
    rows = []
    for ex in items:
        ids = tok(prefix + fmt(ex, False), return_tensors="pt").input_ids.to(dev)
        lp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)[letter_ids]
        pred = int(lp.argmax())
        rows.append({"id": ex["id"], "pred": pred, "gold": int(ex["cop"]), "correct": pred == int(ex["cop"])})
    if out_jsonl:
        open(out_jsonl, "w").write("\n".join(json.dumps(r) for r in rows))
    return rows
