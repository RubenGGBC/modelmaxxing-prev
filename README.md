# Model Maxxing (prev) — spike de cirugía de pesos

Spike de viabilidad: ¿puede un LLM frontera diseñar un parche de pesos que mejore una transferencia de capacidades?
Especificaciones: `plans/` (plan.md = v0.1, MODEL_MAXXING_MVP_v0.2.md, v3.md = siguiente experimento).
Informes: `docs/SPIKE_REPORT.md` (estado y resultados), `docs/BARRIER_REPORT.md` (barrera de interpolación).

## Puesta en marcha
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,dev]"           # torch, safetensors, transformers, datasets, pytest
python scripts/download_models.py      # ~4 GB en models_cache/ (revisiones fijadas)
pytest -q                              # 18 tests, tensores sintéticos
```
Las evaluaciones usan `PYTHONPATH=src`. El runner de MBPP ejecuta código con `sandbox-exec` (solo macOS):
**en Linux/Windows hay que sustituirlo por Docker** (`evaluation/runner.py::run_tests`). El dispositivo es `mps` si existe; cambiar a `cuda` en los scripts (`dev = ...`) para el PC potente.

## Par de modelos (procedencia verificable, ambos derivan de Qwen2.5-0.5B)
- R: Rumiii/Qwen2.5-0.5B-Med-Pre-Trained-92k @915fc914 (CPT PubMed, Apache-2.0)
- D: qgallouedec/Qwen2.5-0.5B-codeforces-SFT @febca969 (licencia no declarada)

## Estado (ver SPIKE_REPORT para el detalle)
Hecho: infraestructura de parches (esquema, validación, store, rollback), transferencia, métricas con bootstrap, barrera de interpolación (real, no un bug), líneas base R/D/B, diagnósticos para el cirujano (`agents/diagnostics.py`), tabla de sensibilidad (`scripts/sensitivity_table.py`), constructores de brazos (`surgery/arms.py`).
Hallazgos clave: la complementariedad del par es débil (R no supera a B ni a D en MedMCQA; en NLL de PubMed sí por ~0.03); la "mejora" de interpolar R→D se reproduce interpolando R→B (es encoger el delta del receptor, no el donante).
**Pendiente (para el PC potente):** generar los parches de los brazos B, B2, C, D (Claude), E con el presupuesto pre-registrado en SPIKE_REPORT (N0=1.0, ≤12 tensores down_proj, rango ≤8), y evaluar cada uno solo y con la transferencia `R*+0.5(D-B)` en MBPP test[:100], MedMCQA val[:300] y NLL PubMedQA[200:260]; incluir el control R→B. Hay que escribir `scripts/run_arms.py` (no existe aún).
