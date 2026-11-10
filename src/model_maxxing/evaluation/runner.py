"""MBPP: generación determinista + ejecución aislada (sandbox-exec: sin red, escritura solo en tmp)."""
from __future__ import annotations

import json
import os
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


def _limits(timeout: int):
    """Límites POSIX del subproceso: memoria 1 GB, CPU, tamaño de fichero, sin core dumps."""
    import resource

    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


_UNSHARE: bool | None = None


def _can_unshare_net() -> bool:
    """¿Podemos aislar la red con `unshare -n` (requiere root, p. ej. en Kaggle/Colab)?"""
    global _UNSHARE
    if _UNSHARE is None:
        try:
            _UNSHARE = shutil.which("unshare") is not None and subprocess.run(
                ["unshare", "-n", "true"], capture_output=True, timeout=10).returncode == 0
        except Exception:
            _UNSHARE = False
    return _UNSHARE


def _run_subprocess_sandbox(d: str, f: Path, timeout: int) -> tuple[bool, str]:
    """Modo MM_SANDBOX=subprocess, SOLO para VMs desechables Linux (Kaggle/Colab) sin Docker:
    subproceso con límites de recursos, directorio temporal, entorno vacío y red cortada si es posible."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("MM_SANDBOX=subprocess solo está soportado en Linux (macOS no admite RLIMIT_AS)")
    cmd = (["unshare", "-n"] if _can_unshare_net() else []) + [sys.executable, "-I", str(f)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           cwd=d, env={"PATH": "/usr/bin:/bin", "HOME": d}, preexec_fn=_limits(timeout))
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if p.returncode in (-9, -24, 137):     # SIGKILL / SIGXCPU por límite de CPU
        return False, "timeout"
    return p.returncode == 0, p.stderr[-300:]


def sandbox_description() -> str:
    if os.environ.get("MM_SANDBOX") == "subprocess":
        return f"subprocess(rlimits, unshare_net={_can_unshare_net()})"
    return "sandbox-exec" if shutil.which("sandbox-exec") else "docker" if shutil.which("docker") else "none"


def run_tests(code: str, ex: dict, timeout: int = 10) -> tuple[bool, str]:
    program = "\n".join(ex.get("test_imports", [])) + "\n" + code + "\n\n" + "\n".join(ex["test_list"]) + "\n"
    if os.environ.get("MM_SANDBOX") == "subprocess":
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "t.py"
            f.write_text(program, encoding="utf-8")
            return _run_subprocess_sandbox(d, f, timeout)
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


def generate_batch(model, tok, dev, prompts: list[str], max_new_tokens: int = 256, stops: list[str] | None = None) -> list[str]:
    """Greedy por lotes con relleno a la izquierda. Determinista para un orden y tamaño de lote fijos,
    pero puede diferir en algún token de la generación de uno en uno: no mezclar ambos modos en un experimento."""
    import torch
    old = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                                 stop_strings=stops or [STOP], tokenizer=tok)
    finally:
        tok.padding_side = old
    n = enc["input_ids"].shape[1]
    return [tok.decode(o[n:], skip_special_tokens=True) for o in out]


def token_budget_chunks(lengths: list[int], batch_size: int, max_new_tokens: int,
                        max_batch_tokens: int | None) -> list[tuple[int, int]]:
    """Corta una lista YA ORDENADA por longitud en lotes [i, j) con como mucho `batch_size` ejemplos y, si hay
    presupuesto, B·(L_max + max_new_tokens) ≤ max_batch_tokens (el relleno a la izquierda iguala todos a L_max).
    Un ejemplo que por sí solo excede el presupuesto va en un lote de 1. Determinista."""
    out, i = [], 0
    while i < len(lengths):
        j = i + 1
        while j < len(lengths) and j - i < batch_size and (
                max_batch_tokens is None or (j - i + 1) * (max(lengths[i:j + 1]) + max_new_tokens) <= max_batch_tokens):
            j += 1
        out.append((i, j))
        i = j
    return out


def evaluate_mbpp_batched(model, tok, dev, examples: list, tag: str, batch_size: int = 16, max_new_tokens: int = 256,
                          on_rows=None, test_workers: int = 4, max_batch_tokens: int | None = None,
                          task=None) -> list[dict]:
    """Genera por lotes (ordenados por longitud de prompt, determinista) y ejecuta los tests en paralelo.
    `max_batch_tokens` limita B·(L_max + max_new_tokens) para no agotar la memoria con prompts muy largos.
    `task` (opcional) cambia el formato: objeto/módulo con build_prompt, extract_code, run(ex, completado) y STOPS
    (p. ej. evaluation.humaneval). Sin `task` el comportamiento es el de MBPP, idéntico al del Hito A.
    `on_rows(rows)` se llama tras cada lote para guardar de forma reanudable."""
    from concurrent.futures import ThreadPoolExecutor
    bp = task.build_prompt if task else build_prompt
    ex_fn = task.extract_code if task else extract_code
    run_fn = task.run if task else (lambda e, c: run_tests(c, e))
    stops = task.STOPS if task else None
    keyed = sorted(((len(tok(bp(e)).input_ids), e["task_id"], e) for e in examples), key=lambda x: x[:2])
    order = [e for _, _, e in keyed]
    chunks = token_budget_chunks([n for n, _, _ in keyed], batch_size, max_new_tokens, max_batch_tokens)
    mode = f"batch{batch_size}" + (f"_tok{max_batch_tokens}" if max_batch_tokens else "")
    all_rows = []
    with ThreadPoolExecutor(test_workers) as pool:
        for i, j in chunks:
            chunk = order[i:j]
            prompts = [bp(e) for e in chunk]
            t0 = time.time()
            texts = generate_batch(model, tok, dev, prompts, max_new_tokens, stops)
            dt = (time.time() - t0) / len(chunk)
            results = list(pool.map(lambda a: run_fn(a[0], ex_fn(a[1])), zip(chunk, texts)))
            rows = [{"task_id": e["task_id"], "model": tag, "prompt": p, "generation": t, "passed": ok, "error": err,
                     "gen_seconds": round(dt, 2), "gen_mode": mode}
                    for e, p, t, (ok, err) in zip(chunk, prompts, texts, results)]
            if on_rows:
                on_rows(rows)
            all_rows += rows
    return all_rows


def evaluate_mbpp(model_path: str, examples: list, out_jsonl: str, max_new_tokens: int = 256) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = best_device()
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).to(dev).eval()
    return evaluate_mbpp_model(model, tok, dev, examples, out_jsonl, model_path, max_new_tokens)
