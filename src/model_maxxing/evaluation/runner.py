"""MBPP: generación determinista + ejecución aislada (sandbox-exec: sin red, escritura solo en tmp)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

PROFILE = '(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (subpath "{d}"))'
STOP = "```"


def build_prompt(ex: dict) -> str:
    tests = "\n".join(ex["test_list"])
    return f"{ex['prompt']}\nYour code should pass these tests:\n{tests}\n\n```python\n"


def extract_code(text: str) -> str:
    return text.split(STOP)[0]


def _sandbox_cmd(d: str, f: Path, timeout: int, name: str) -> list[str]:
    """Aislamiento obligatorio: sandbox-exec (macOS) o Docker sin red. Si no hay ninguno, NO se ejecuta.
    En Docker el límite de tiempo se aplica dentro del contenedor (el arranque no cuenta)."""
    if shutil.which("sandbox-exec"):
        return ["sandbox-exec", "-p", PROFILE.format(d=d), sys.executable, "-I", str(f)]
    if shutil.which("docker"):
        return ["docker", "run", "--rm", "--name", name, "--network", "none", "--memory", "512m", "--cpus", "1",
                "--pids-limit", "64", "--read-only", "-v", f"{d}:/w:ro", "python:3.11-slim",
                "timeout", "-s", "KILL", str(timeout), "python", "-I", "/w/t.py"]
    raise RuntimeError("Sin sandbox (sandbox-exec/docker): no se ejecuta código generado")


def run_tests(code: str, ex: dict, timeout: int = 10) -> tuple[bool, str]:
    program = "\n".join(ex.get("test_imports", [])) + "\n" + code + "\n\n" + "\n".join(ex["test_list"]) + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(program, encoding="utf-8")
        name = f"mbpp-{uuid.uuid4().hex[:12]}"
        cmd = _sandbox_cmd(d, f, timeout, name)
        outer = timeout if cmd[0] != "docker" else timeout + 60
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=outer, cwd=d)
        except subprocess.TimeoutExpired:
            if cmd[0] == "docker":
                subprocess.run(["docker", "kill", name], capture_output=True)
            return False, "timeout"
        if cmd[0] == "docker" and p.returncode in (124, 137):
            return False, "timeout"
        return p.returncode == 0, p.stderr[-300:]


def best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def evaluate_mbpp_model(model, tok, dev, examples: list, out_jsonl: str, tag: str = "", max_new_tokens: int = 256) -> list[dict]:
    import torch
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
            row = {"task_id": ex["task_id"], "model": tag, "prompt": prompt, "generation": text,
                   "passed": ok, "error": err, "gen_seconds": round(time.time() - t0, 2)}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
    return rows


def evaluate_mbpp(model_path: str, examples: list, out_jsonl: str, max_new_tokens: int = 256) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = best_device()
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).to(dev).eval()
    return evaluate_mbpp_model(model, tok, dev, examples, out_jsonl, model_path, max_new_tokens)
