# Obsoleto: pass@1 local con repetition_penalty heredado

Los 238 problemas de `pass_B.jsonl` se generaron cargando el esqueleto de R (Qwen2.5-1.5B-Instruct), cuya
`generation_config` trae `repetition_penalty=1.1`. Ese valor se aplica incluso en greedy (`do_sample=False`).
No era el protocolo decidido, así que estos resultados **no se usan**.

Corregido en `models/loading.py::neutral_generation_config` (greedy puro, penalización 1.0, EOS fijos).
Las NLL de esta carpeta (`nll_*.jsonl`) siguen siendo válidas: solo usan el forward, no la generación.
