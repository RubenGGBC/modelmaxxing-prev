"""MBPP: generación determinista + ejecución aislada (sandbox-exec: sin red, escritura solo en tmp)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROFILE = '(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (subpath "{d}"))'
STOP = "```"


def build_prompt(ex: dict) -> str:
    tests = "\n".join(ex["test_list"])
    return f"{ex['prompt']}\nYour code should pass these tests:\n{tests}\n\n```python\n"


def extract_code(text: str) -> str:
    return text.split(STOP)[0]


def run_tests(code: str, ex: dict, timeout: int = 10) -> tuple[bool, str]:
    program = "\n".join(ex.get("test_imports", [])) + "\n" + code + "\n\n" + "\n".join(ex["test_list"]) + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(program)
        cmd = ["sandbox-exec", "-p", PROFILE.format(d=d), sys.executable, "-I", str(f)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return p.returncode == 0, p.stderr[-300:]


def evaluate_mbpp(model_path: str, examples: list, out_jsonl: str, max_new_tokens: int = 256) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(dev).eval()
    rows = []
    with open(out_jsonl, "w") as fh:
        for ex in examples:
            prompt = build_prompt(ex)
            ids = tok(prompt, return_tensors="pt").to(dev)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id, stop_strings=[STOP], tokenizer=tok)
            text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            ok, err = run_tests(extract_code(text), ex)
            row = {"task_id": ex["task_id"], "model": model_path, "prompt": prompt, "generation": text,
                   "passed": ok, "error": err, "gen_seconds": round(time.time() - t0, 2)}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
    return rows
