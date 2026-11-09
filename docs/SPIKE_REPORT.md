# Informe del spike — Model Maxxing (ESTADO: BLOQUEADO, sin resultados)

## Hecho
- Estructura del repo, `pyproject.toml`, `configs/spike.yaml`.
- Implementados: comprobación de compatibilidad (`models/compatibility.py`), interpolación, task vector con máscaras, estadísticas de deltas.
- Tests mínimos (identidad de interpolación y delta, máscara vacía, incompatibilidad, NaN/Inf, reproducibilidad): 6/6 pasan con tensores sintéticos.

## No realizado (no hay resultados)
- Compatibilidad real de checkpoints, baselines (MBPP, médico), barridos de interpolación/delta, decisión Go/Pivot/Stop.

## Bloqueos (el plan exige documentarlos y preguntar antes de sustituir)
1. **Receptor:** `LOCAL_MEDICAL_QWEN_CHECKPOINT` no existe en esta máquina. Hace falta ruta o repo HF y su procedencia.
2. **Hardware:** Mac M4 con 16 GB RAM y ~16 GB de disco libre. Tres modelos 3B en fp16 (~6 GB cada uno) no caben en disco, y la evaluación de generación en MPS/CPU será lenta. Opciones: GPU remota, liberar disco, o pivotar a un par de ~0.5B (Qwen2.5-0.5B / Coder-0.5B).
3. **Entorno aislado** para ejecutar código de MBPP (Docker o similar): no verificado.
4. **Benchmark médico:** sin elegir (MedQA u otro) y sin comprobar solapamiento con el ajuste del receptor.

## Pendiente de implementar
`evaluation/runner.py`, `metrics.py` (bootstrap pareado), scripts de barrido y `experiments/pipeline.py`.

## Dato real obtenido (exploratorio, par 0.5B, no es el par del plan)
- Qwen2.5-0.5B vs Qwen2.5-Coder-0.5B: mismos tensores y formas, misma config de arquitectura y tokenizer idéntico (hash de tokenizer.json). Salida en `results/reports/compatibility.json` (la columna "receiver" ahí es la base repetida, no un receptor médico real).
- La procedencia (Coder derivado de la base Qwen2.5) NO está verificada aquí; el task vector sigue siendo exploratorio hasta comprobarlo con la model card.
- Añadidos `evaluation/metrics.py` (bootstrap pareado) y su test: 8/8 tests pasan.

## Experimento A parcial (exploratorio, par 0.5B, SIN receptor médico ni baseline médico)
- MBPP sanitized, primeros 100 de `test`, greedy, 256 tokens, prompt fijo, fp16 en MPS. Resultados por problema en `results/metrics/mbpp_*.jsonl`.
- Base 0.5B: 47/100. Coder 0.5B: 55/100. Diferencia +8 pp, IC95% bootstrap pareado [-1, +17] pp; 15 problemas pasan de mal a bien y 7 de bien a mal.
- Lectura: el Coder tiende a ser mejor, pero el IC incluye 0. H1 no queda establecida con esta muestra.
- Aislamiento: `sandbox-exec` (sin red, escritura solo en tmp) + timeout de 10 s. Docker está instalado pero su daemon no estaba arrancado. Es más débil que un contenedor y no limita la memoria.
- Esto no responde H2 (falta el receptor y el benchmark médico). No hay decisión Go/Pivot/Stop.

## Barrido de interpolación (PAUSADO; receptor sustituto = base 0.5B, solo prueba de pipeline)
MBPP[:100], greedy. α=0.00: 47 (el pipeline reproduce la base). α=0.25: 17. α=0.50: 2. α=0.75: 17. α=1.00 por el pipeline: NO verificado (ejecución detenida a petición del usuario); el Coder directo dio 55.
Sin verificar α=1.0, la caída en α intermedios no se puede atribuir con certeza a la mezcla y no a un bug.

## Actualización v0.2 (MODEL_MAXXING_MVP_v0.2.md sustituye al plan.md)
- Implementada la capa de cirugía: `surgery/` (esquema, builder, validador, store con hash), `transfer/engine.py`, `agents/frontier_surgeon.py` (inspección de regiones y parseo de propuestas JSON sin ejecutar código).
- Tests (tensores sintéticos): parche no nulo, cotas de norma/lista permitida/finitud, persistencia tras interpolación (efecto = (1-α)·P, desaparece en α=1), rollback, originales intactos, ida y vuelta del store, baselines. 15/15 pasan.
- Los experimentos A–E con modelos reales NO se han ejecutado. Los resultados de interpolación de arriba siguen sin validar (α=1.0 pendiente).

## Validación del barrido de interpolación (COMPLETA)
Por el mismo pipeline: α=0 → 47/100 (= base), α=1 → 55/100 (= Coder directo). La caída en intermedios (α=0.25: 17, 0.5: 2, 0.75: 17) es real, no un bug del pipeline. Coherente con delta Coder−Base de norma 34.2 vs 38.8 de los pesos en layers.12.mlp.down_proj (88% relativo): la interpolación lineal cae en una barrera entre ambos. Datos reales sobre el par 0.5B, receptor sustituto = base (sin medicina).

## Puerta 2 (cirujano): parche 1367e530f23585cf
Rango 1, bloque 16x16 en layers.12.mlp.down_proj (480,384), dirección singular dominante del delta del bloque a 0.5·σ1; ||P||=0.144 (0.37% del tensor). Validado y guardado. Experimento A/C/D/E con α=0.1 y N=100 fijados antes de ejecutar: ver results/metrics/surgery_experiment.json.

## Resultado A/C/D/E (par 0.5B, receptor sustituto = base, MBPP[:100], α=0.1 fijado antes)
| Variante | Aciertos |
|---|---|
| A Transfer(R,D) | 44 |
| C Transfer(R+P_Claude,D) | 44 |
| D Transfer(R+P_control,D) | 44 |
| E R+P_Claude (sin transferencia) | 46 |
| R+P_control (sin transferencia) | 47 |
| R (base) | 47 |

- Persistencia: ||W_final(P)-W_final(0)|| = 0.130 = (1-α)·||P|| (0.9·0.1447), como predice la teoría. Control: 0.1298. R* ≠ R verificado.
- C vs A y C vs D: diferencia 0, sin problema que cambie. E vs R: -1 problema (IC95% [-3, 0] pp), un cambio correcto→incorrecto.
- Conclusión: H3 (parche no trivial, reproducible, persistente y reversible) se demuestra técnicamente. H4 NO se sostiene: el parche de 16x16 (0.37% del tensor) no produce efecto medible frente a A ni al control. Resultado nulo esperable con un bloque tan pequeño; no descarta cirugías mayores. Medicina/protección no evaluadas (no hay receptor ni benchmark médico), así que no hay decisión Go/Pivot/Stop de investigación.
- Decisión provisional: GO técnico parcial (items 2–4 del criterio: operador, parche aplicado/persistente/revertible, reproducible), pero sin complementariedad real demostrada (falta H1/H2). PIVOT sugerido: regiones mayores y señales mejores (p. ej. varias capas, subespacios de activaciones) con los mismos controles.

## v0.3 Prioridad 1 completada
Ver `docs/BARRIER_REPORT.md`: barrera de pérdida real (NLL código 1.39→2.44→1.30, no implementación). Checkpoints en bf16 (corrige "fp16" anterior). Embeddings con diferencia relativa 1.02 entre Base y Coder.

## v0.3 pre-registro (escrito ANTES de generar parches)
Par con procedencia: B=Qwen2.5-0.5B, R=Rumiii/Qwen2.5-0.5B-Med-Pre-Trained-92k @915fc914 (CPT PubMed, Apache-2.0), D=qgallouedec/Qwen2.5-0.5B-codeforces-SFT @febca969 (SFT completo, licencia no declarada). Ambos derivan de B.
Presupuesto: N0=1.0 (norma Frobenius total), <=12 tensores down_proj, rango <=8. Transferencia posterior: R*+0.5*(D-B). Tolerancia: MedMCQA >= R-5pp; NLL bio test <= R+0.02.
Visible para el agente: MBPP full train, PubMedQA[100:160], MedMCQA train. Evaluación: MBPP sanitized test[:100], MedMCQA val[:300], PubMedQA[200:260].

## v0.3 Experimento decisivo: brazos A–E (2026-11-09, RTX 3060 Ti, FP32)
Pre-registro commiteado antes de evaluar (`6640786`): `configs/arm_D_claude_design.json` y hashes de los parches en `results/arms/manifest.json`.
Evaluación: MBPP sanitized test[:100], MedMCQA val[:300] 3-shot, NLL bio en PubMedQA[200:260], NLL de código en las soluciones de MBPP test[:100].
Todos los parches tienen norma 1,000 y están aplicados en FP32. Transferencia: R* + 0,5·(D−B). Resultados por problema en `results/arms/`.

| Config | MBPP | MedMCQA | NLL bio | NLL código | Válido |
|---|---|---|---|---|---|
| A (R) | 38 | 0,313 | 2,2991 | 1,2996 | — |
| A_T | 45 | 0,303 | 2,2986 | 1,3006 | sí |
| B (SVD Δ_D) | 43 | 0,320 | 2,3425 | 1,4012 | **no** (bio +0,043) |
| B_T | 45 | 0,303 | 2,3682 | 1,4444 | **no** |
| B2 (SVD −Δ_R) | 37 | 0,313 | 2,2986 | 1,2933 | sí |
| B2_T | 45 | 0,313 | 2,2975 | 1,2966 | sí |
| C (regla fija) | 41 | 0,310 | 2,2844 | 1,2869 | sí |
| C_T | 43 | 0,313 | 2,2853 | 1,2906 | sí |
| **D (Claude)** | 41 | 0,317 | 2,2853 | 1,2840 | sí |
| **D_T** | 43 | 0,320 | 2,2863 | 1,2882 | sí |
| E (aleatorio ≈ D) | 39 | 0,313 | 2,2990 | 1,2999 | sí |
| E_T | 43 | 0,307 | 2,2984 | 1,3006 | sí |
| RB_05 (0,5R+0,5B, sin parche) | **49** | **0,337** | **2,2719** | **1,2440** | sí |

Comparaciones clave (IC95% bootstrap pareado):

| Comparación | MBPP | MedMCQA | ΔNLL bio | ΔNLL código |
|---|---|---|---|---|
| D vs E | +2 pp [−3, +7] | +0,3 pp [−1,7, +2,7] | −0,0137 | −0,0159 |
| D vs C | +0 pp [−3, +3] | +0,7 pp [−0,7, +2,0] | +0,0009 | −0,0028 |
| D_T vs A_T | −2 pp [−7, +3] | +1,7 pp [−0,7, +4,3] | −0,0123 | −0,0125 |
| D_T vs E_T | +0 pp [−5, +5] | +1,3 pp [−1,0, +3,7] | −0,0121 | −0,0124 |
| A_T vs A | +7 pp [+0, +14] | −1,0 pp [−2,7, +0,7] | −0,0004 | +0,0010 |
| RB_05 vs A | **+11 pp [+3, +19]** | +2,3 pp [−1,0, +6,0] | −0,0271 | −0,0556 |

### Lectura
1. **La dirección de D es real.** Con la misma norma, el control aleatorio E no mueve la NLL (±0,0003), mientras que D la baja unos 0,014 en bio y 0,016 en código, tanto solo como tras la transferencia. Se cumplió la predicción pre-registrada (0,01–0,02 nats).
2. **No hay aportación atribuible a Claude.** D y C (regla fija sobre la misma tabla) apuntan casi en la misma dirección (coseno 0,89) y dan lo mismo: 0 pp en MBPP y diferencias de NLL ≤ 0,003. La información útil está en la tabla de sensibilidad, no en el diseño.
3. **MBPP no separa ningún parche** (todas las diferencias entre brazos caen dentro del ruido de ±9 pp). Los parches no suman nada a la transferencia: D_T, C_T y E_T empatan a 43 frente a 45 de A_T.
4. **El SVD convencional del donante (B) es dañino:** NLL de código +0,10 y bio fuera de tolerancia. Concentra la norma en las capas 12–23, justo donde la tabla visible marcaba dD+ como perjudicial.
5. **El confusor domina.** Encoger el receptor hacia la base sin cirugía (RB_05) es lo único significativo en MBPP (+11 pp, IC [+3, +19]), mejora MedMCQA y baja la NLL de 2 a 4 veces más que cualquier parche. Con N0 = 1 frente a ‖Δ_R‖ = 14, el presupuesto de cirugía es demasiado pequeño para competir con ese efecto.
6. Por el criterio pre-registrado, el mejor válido en MBPP es un empate a 45 entre A_T y B2_T. Ningún parche supera a la transferencia sin cirugía.

### Decisión: **PIVOT** (no Stop, no Go)
- **No es Go:** no hay evidencia de que Claude aporte una modificación de pesos útil frente a una regla fija con la misma información.
- **No es Stop:** la infraestructura funciona de punta a punta y las direcciones guiadas por sensibilidad tienen un efecto real y medible en NLL, distinto del aleatorio. Lo que falla es el par de modelos (complementariedad débil, receptor sobreajustado) y la escala del presupuesto.
- **Pivote propuesto:**
  - Usar como referencia un receptor regularizado (p. ej. RB_05) para quitar el confusor.
  - Escalar N0 hasta que el efecto supere la resolución de MBPP.
  - Dar a Claude información que la regla fija no tenga (gradientes, activaciones por ejemplo), para que el diseño pueda diferenciarse.
  - Medir con MBPP completo (257) y varias semillas de E.

Limitaciones: 100 problemas MBPP, una sola semilla del control E, un único par de modelos de 0,5B. Las cifras de B/R/D ya existentes (Mac, fp16) difieren ligeramente de A (CUDA, fp32): R da 38 frente a 39.
