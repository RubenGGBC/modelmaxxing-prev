# Baseline v0.3 — BLOQUEADO (decisión: PIVOT)

Referencia inmutable del estado al cerrar v0.3. **No editar los resultados aquí listados.** Cualquier experimento nuevo va en `results/experiments/` y documentos nuevos.

- **Tag git:** `v0.3-pivot` → commit `10fb172` ("Resultados del experimento de brazos A-E e informe final (PIVOT)").
- **Pre-registro de brazos:** commit `6640786` (diseño de D y manifiesto, commiteados antes de evaluar).
- **Informe completo:** [SPIKE_REPORT.md](SPIKE_REPORT.md) (sección "v0.3 Experimento decisivo"). Barrera de interpolación: [BARRIER_REPORT.md](BARRIER_REPORT.md).

## Decisión

**v0.3 = PIVOT.** El parche de Claude (D) altera probabilidades de forma real (baja NLL frente al aleatorio E), pero **no hay ventaja atribuible a Claude** frente a la regla fija C con la misma información, y ningún parche mejora a la transferencia sin cirugía. El control de vuelta a la base (`0,5·R + 0,5·B`) domina todo. Esa mejora es **recuperación del sobreajuste del receptor**, no transferencia desde el donante.

| Variante | MBPP / 100 | NLL código | Lectura |
|---|---:|---:|---|
| R original (A) | 38 | 1,2996 | Baseline |
| Transferencia sin cirugía (A_T) | 45 | 1,3006 | Control obligatorio |
| Claude sin transferencia (D) | 41 | 1,2840 | Efecto real en NLL |
| Claude + transferencia (D_T) | 43 | 1,2882 | No mejora a A_T |
| Regla fija (C) | 41 | 1,2869 | Empata con Claude |
| `0,5·R + 0,5·B` (RB_05) | **49** | **1,2440** | Control de retorno a base, dominante |

## Modelos (Qwen2.5-0.5B, bf16 en disco, evaluados en FP32)

| Rol | Repo | Revisión | sha256 `model.safetensors` |
|---|---|---|---|
| B | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| R | `Rumiii/Qwen2.5-0.5B-Med-Pre-Trained-92k` | `915fc914926974078c123044cefbf9c12d639877` | `a6ea162894de0a257c2629e7c2e30c425819b95fcce259834c43d914a89dc522` |
| D | `qgallouedec/Qwen2.5-0.5B-codeforces-SFT` | `febca9694f5a908e37199740cb92bfe3d70895a1` | `7087feb80acacf3355b655d5d144d498fb3ea0d652449614740bf41cc8fcfc26` |
| (exploratorio) | `Qwen/Qwen2.5-Coder-0.5B` | `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301` | `aff8914ec707fcaf9e2d4dc97197cded50b1c63e1d3a7a82e56f54d83ea47f80` |

Ruta local: `models_cache/` (no versionada). Descarga reproducible: `scripts/download_models.py`.

## Artefactos y hashes

| Artefacto | sha256 |
|---|---|
| `configs/arm_D_claude_design.json` | `93a87dd920cd5bc8437694e108f031654447b9411b73054f48aec972ded0cd96` |
| `results/arms/manifest.json` | `be2ce2e3d58f222859e50ccbdde7ac6ae3af2ae5d74ba83733d44e7fdf3d44e4` |
| `results/arms/report.json` | `0617d3cab8c541c06d72cff93f358571ff3e30ed99e278c4bbda8ecd961c51bf` |

Parches (`results/arms/patches/`, 73 MB, **no versionados** por `.gitignore`; regenerables con `scripts/run_arms.py`):

| Brazo | patch_id | Autor | sha256 del JSON |
|---|---|---|---|
| B | `00c7ef4cfb8e34b1` | SVD de Δ_D | `f7d450efa9d29555402b0793cd0b355bd2468e3073c241bb0f7f5eaf0d83dfc6` |
| B2 | `ae7d9955b1a0133b` | SVD de −Δ_R | `d2a8116229dcbd2718e284c1b946c5027573a8b5f8d1d45df58f5579b6e9ccd4` |
| C | `881fd39c242758df` | Regla fija | `0aaba8608a8b52c0b1ccdff6b1a9f12177f842722569023aa030570d6d867580` |
| D | `80dbfc7b3bff5e7d` | Claude | `a9bfa66247c16cbb56ac00402af2d58e0a6e0534322be4bc86e0984d3a5f2235` |
| E | `9f0a51376375074e` | Aleatorio igualado a D | `15757538a8ebfdf920d40d10fc45f44ef7439206f1c6f63c861122bfe7eea72d` |

Resultados por problema: `results/arms/{mbpp,med}_*.jsonl`, resúmenes `summary_*.json`, log `run.log`. Diagnósticos visibles para el agente: `results/metrics/v3_*.json`, `visible_mbpp_*.jsonl`.

## Entorno de ejecución (registrado 2026-11-09)

| Componente | Versión |
|---|---|
| SO | Windows 11 Home 10.0.26300 |
| GPU | NVIDIA GeForce RTX 3060 Ti, 8192 MiB, driver 621.01 |
| Python (`.venv`) | 3.11.7 |
| PyTorch | 2.6.0+cu124 (CUDA 12.4) |
| transformers / datasets | 5.22.2 / 5.1.0 |
| Docker (sandbox MBPP) | 29.6.1 |
| RAM | 15,8 GB |
| Tests | `pytest -q` → **18 passed** |

## Datos ya consumidos — NO son test final en v0.4

- **MBPP sanitized `test[:100]`**: evaluado repetidamente en v0.2 y v0.3. Prohibido como test final; prohibido reoptimizar sobre él.
- MBPP `train` completo: visible para el agente en v0.3 (puede reutilizarse como diagnóstico).
- MedMCQA `val[:300]`, PubMedQA `[100:160]` y `[200:260]`: usados en v0.3.

## Limitaciones conocidas

- 100 problemas MBPP (IC95% ≈ ±9 pp): MBPP no separa ningún parche.
- Una sola semilla del control aleatorio E.
- Un único par 0,5B, con complementariedad débil y receptor sobreajustado (‖Δ_R‖ = 14,0 frente a ‖Δ_D‖ = 1,77).
- Presupuesto de cirugía N0 = 1,0, muy pequeño frente a ‖Δ_R‖.
- Las cifras antiguas de B/R/D (Mac, fp16) difieren ligeramente de las de A (CUDA, fp32): R da 38 frente a 39.
- Licencia del donante (`codeforces-SFT`) no declarada: no redistribuir derivados.
