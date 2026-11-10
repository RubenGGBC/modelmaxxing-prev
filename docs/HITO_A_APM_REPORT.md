# Hito A — Viabilidad de APM (Qwen2.5-7B → Qwen2.5-1.5B)

**Fecha:** 10-11-2026 · **Máquina:** MacBook con Apple M4 (16 GB unificados) · **Commit de partida:** `dbd633b`

## 0. Resumen y decisión

| Pregunta | Respuesta |
|---|---|
| ¿Está APM implementado fielmente? | **Sí**, para Qwen2: perfilado, selección APM, línea base IM y fusión. 8 tests nuevos, incluida la equivalencia exacta del perfilado capa a capa con el forward completo de HF. |
| ¿Es 7B→1.5B compatible con APM? | **Sí.** Mismas 28 capas y head_dim 128; el donante es mayor en todas las dimensiones. Hay dos diferencias, tratadas en el código: el vocabulario (152 064 frente a 151 936) y los embeddings (lm_head propio en el 7B, atados en el 1.5B). |
| ¿Cabe en este hardware? | **Cómputo y RAM: sí**, porque el 7B se procesa capa a capa (≈0,5 GB por capa). **Disco: no.** Hacen falta ≈29,8 GB libres y hay entre 9 y 13 GB. |
| ¿Funciona el circuito completo? | **Sí**, con pesos reales de 0.5B: perfil → slice → fusión → pass@1 en sandbox → congelación de μ → test → informe. |
| ¿Se ejecutó el experimento R / IM / APM con 7B→1.5B? | **No.** Ni el 7B (15,23 GB) ni el 1.5B (3,09 GB) están descargados, y no caben. No se lanzó ninguna descarga. |
| ¿Mejora APM al receptor? | **Sin evidencia.** No hay ningún resultado de 1.5B en este hito. |

### Decisión: **GO técnico, condicionado a disco**. No hay ningún GO científico.

- **Por qué no es STOP:** la implementación funciona, el par es compatible y la máquina tiene cómputo y RAM suficientes. El coste estimado (§7) es asumible: unos minutos de perfilado y entre 3 y 7 h de evaluación.
- **Por qué no es PIVOT:** nada de lo observado obliga a cambiar el método ni el par. El único bloqueo es de espacio en disco.
- **Condición para ejecutar:** **≥ 30 GB libres** en este Mac, o ejecutar en otra máquina (el PC con RTX 3060 Ti o Kaggle) con la misma configuración. La fase `resources` lo comprueba y falla de forma controlada si no se cumple. **La ejecución en Kaggle está preparada (§7.1).**

---

## 1. Auditoría del repositorio

**Módulos existentes que se reutilizan**

| Módulo | Uso en el Hito A |
|---|---|
| `models/loading.py` (`CheckpointReader`, `materialize`, `Recipe`) | Lectura perezosa de safetensors y fusión en FP32. **La fusión de APM, `(1−μ)θ_s + μθ_slice`, es exactamente la transferencia `interpolate` existente** con el slice como donante: no se ha duplicado código. |
| `evaluation/splits.py` | Particiones pre-registradas, disjuntas y con el conjunto quemado de v0.3. |
| `evaluation/runner.py` | Generación greedy por lotes y ejecución de tests en sandbox (`sandbox-exec` en macOS, Docker en Linux/Windows). |
| `evaluation/metrics.py` | Bootstrap pareado por problema. |
| `surgery/*`, `agents/*`, `transfer/{task_vector,interpolation,masks}` | No se usan en este hito (cirugía/Claude: fuera de alcance). |

**Tests:** al empezar, 25 pasaban y 2 fallaban. Los 2 fallos eran del sandbox por subproceso, que es solo para Linux: el `skipif` usaba `os.name != "posix"`, pero macOS también es POSIX y no admite `RLIMIT_AS`. Se corrigió la condición en el test y en la guarda del runner (`sys.platform`). **Ahora: 33 passed, 2 skipped** (los 2 omitidos son los de Linux).

**v0.3:** archivado como PIVOT en `docs/BASELINE_V03_LOCKED.md` (tag `v0.3-pivot`). No se ha tocado.

**Discrepancia que conviene conocer.** El historial contiene commits titulados «Hito A v0.4» (`73145e1`, `3797eaf`, `dbd633b`). Corresponden a la selección de un par **1.5B-Instruct / Coder-1.5B del plan v0.4 anterior**, ejecutada en el PC con RTX 3060 Ti o en Kaggle. Solo hay NLL en la partición de selección; el pass@1 se anuló por un `repetition_penalty` heredado. **No son pruebas de este Hito A (APM) y no se han usado.** Su única consecuencia es que la partición `selection` ya tiene NLL de modelos 1.5B, lo que no impide usarla como desarrollo. **El test `final_mbpp` sigue sin tocarse.**

## 2. Hardware y entorno (medido)

| Recurso | Valor | Implicación |
|---|---|---|
| CPU/GPU | Apple M4, 10 núcleos, GPU por **MPS**; **sin CUDA** | Todo en MPS. Sin float64 en GPU (estadísticas acumuladas en CPU). |
| RAM | 16 GB unificados (3,7 GB disponibles con las apps abiertas) | El 7B no cabe entero; **capa a capa sí**. Para evaluar el 1.5B en FP32 (≈6,2 GB) conviene cerrar aplicaciones. |
| Disco | 228 GB, **95-96 % ocupado, 8,8-13,4 GB libres** (varía con el swap) | **Bloqueante.** |
| Python / torch / transformers | 3.14.7 / 2.15.0 / 5.22.2 (datasets 5.1.0, safetensors 0.8.0) | Distinto del entorno de v0.3 (Windows, Python 3.11, torch 2.6+cu124). |
| Sandbox de benchmarks | `sandbox-exec`: **operativo** (sin red, escritura solo en tmp). Docker instalado, demonio **parado**. | Suficiente en macOS. |
| Datos | MBPP en caché local (offline OK). **HumanEval+ no está en caché.** | Test = MBPP; HumanEval+ queda pendiente (§5). |

## 3. APM: algoritmo publicado y nuestra implementación

**Fuente:** Fan et al., *Training-Free Knowledge Transfer Across Model Scales through Activation-Guided Pruning* (arXiv 2608.13596, jul. 2026). El paper usa Qwen2.5-32B→3B (y 14B en fusión secuencial). **No publica 7B→1.5B ni código.**

| Paso | Paper | Implementación (`transfer/apm.py`, `apm_profile.py`) |
|---|---|---|
| Estadísticas (Eq. 3, 9-12) | Media sobre tokens perfilados: capa ‖h_r‖₂, canal \|h_c\|, cabeza ‖o_k‖₂, neurona \|a_j\| | Igual, en streaming, sin guardar tensores de activación |
| Capas (Eq. 4) | TopK, en orden del donante | Igual. Con 28 = 28 capas se cogen todas, y APM = IM en este eje. |
| Canales ocultos (Eq. 5) | TopK global, compartido por embeddings, normas y proyecciones | Igual |
| Cabezas con GQA (Eq. 6) | TopK de Q por capa; KV por la suma de las Q de su grupo, elegidas aparte | Igual (Q para q/o, KV para k/v, sesgos incluidos) |
| MLP (Eq. 5) | TopK por capa, mismos índices en gate/up/down | Igual |
| Fusión (Eq. 8) | θ = (1−μ)θ_s + μθ_slice, μ ∈ (0, 0,10); probados 0,01-0,15 | Igual, en FP32 |
| IM (Eq. 1-2) | Primeras capas y primeros índices, misma fusión | Igual |

**Decisiones nuestras donde el paper no especifica** (todas documentadas en el código):

1. **Canal oculto:** |h| promediado sobre las salidas de todas las capas.
2. **«Post-activación» en SwiGLU:** `act(gate)·up`, es decir, la entrada de `down_proj`.
3. **Ámbito del perfilado:** solo el prompt. Es una de las opciones que da el paper, que no dice cuál usó en código. `prompt+reference` está implementado como variante.
4. **Tamaño de calibración:** 128 ejemplos. El paper no lo fija para sus resultados principales; probó entre 25 y 1000.
5. **μ:** rejilla {0,01; 0,02; 0,05; 0,10}, elegida en dev. El μ de la tabla principal del paper no está publicado.
6. **Evaluación en FP32.** Con μ = 0,01, la perturbación relativa (~1 %) es del orden del paso de redondeo de bf16 (2⁻⁸ ≈ 0,4 %), así que en bf16 la fusión se cuantizaría. El paper no indica precisión.

**Incompatibilidades y rarezas de 7B→1.5B (no impiden ejecutar):**

- **Vocabulario:** se toman las primeras 151 936 filas del embedding del 7B. Los ids coinciden en Qwen2.5; las 128 filas sobrantes son relleno.
- **lm_head:** el 1.5B tiene embeddings atados, así que el lm_head del 7B **no se usa** y el embedding fusionado hace de lm_head.
- **GQA:** el grupo es de 7 Q por KV en el 7B y de 6 en el 1.5B. Al elegir Q y KV por separado (como el paper), una cabeza Q puede acabar emparejada con una cabeza KV distinta de la suya.
- **Normalización:** sin reescalar la RMSNorm al quitar el 57 % de los canales (el paper tampoco lo hace).
- **Coordenadas no alineadas (riesgo científico, no técnico).** El 7B y el 1.5B se entrenaron por separado: el canal *i* del 7B no tiene relación con el canal *i* del 1.5B. La fusión con μ pequeño suma un slice estructurado pero no alineado. Por eso hace falta un **control aleatorio** de la misma forma (§8) antes de atribuir cualquier mejora a la «selección guiada».

**Memoria (calculada para el par real):**

| Fase | Pico estimado | Disco |
|---|---|---|
| Perfilado del 7B capa a capa (bf16) | embedding 1,09 GB (se libera) + 0,47 GB/capa + estados ocultos ≈0,1 GB → **< 2 GB** | 7B: 15,23 GB |
| Construcción del slice | slice 3,09 GB (bf16) + tensor temporal ≤1,1 GB → **≈4-5 GB de RAM** | 2 slices × 3,09 GB |
| Evaluación del 1.5B en FP32 | ≈6,2 GB + caché KV | 1.5B: 3,09 GB |
| **Total de disco** | | **≈24,6 GB + 5 GB de margen = 29,8 GB** |

## 4. Variantes preparadas

| Variante | Definición | Estado |
|---|---|---|
| **R** | Qwen2.5-1.5B original (μ = 0) | Configurada |
| **IM** (recorte fijo) | Slice front-aligned del 7B, fusionado con μ | Implementada y probada |
| **APM** | Slice guiado por activaciones del 7B, fusionado con μ | Implementada y probada |
| RAND (opcional) | Slice de la misma forma con índices aleatorios | Implementada (`random_selection`); **fuera del pre-registro** de 3 brazos |

## 5. Diseño de evaluación (pre-registrado en `configs/hito_a_apm.yaml`)

| Partición | Origen | n | Uso | ids sha256[:16] |
|---|---|---:|---|---|
| Calibración | MBPP full train/prompt, `diagnostic[0:128]` | 128 | Perfilado de APM | `8d1e3a4917a54126` |
| Sonda | `diagnostic[256:320]` | 64 | NLL de slices (diagnóstico) | `e06c7bacb21dc9ec` |
| **Desarrollo** | MBPP full validation + test fuera de sanitized | 333 | Elegir μ | `0d2e406f0d4ba0c8` |
| **Test** | MBPP sanitized `test[100:]` | 157 | Comparación final, **sellada hasta `lock`** | `af71c4994f4bd364` |
| Quemado | MBPP sanitized `test[:100]` (v0.3) | 100 | **No se usa** | — |

Se verifica automáticamente que son disjuntas entre sí y con el conjunto quemado (`check_partitions`, `check_disjoint`).

- **Métrica primaria:** pass@1 greedy (512 tokens, generation_config neutra, sandbox). **Auxiliar:** NLL de la solución de referencia.
- **Regla de μ (fijada antes de ver datos):** por método, el μ con mayor pass@1 en dev; si empatan, menor NLL; si siguen empatados, menor μ. La fase `lock` guarda la elección con hashes de slices y configuración, y **la fase `test` no arranca sin ese fichero**.
- **Efecto mínimo de interés:** +3 pp en test frente a R y frente a IM, con IC95 % del bootstrap pareado > 0.
- **Potencia (advertencia honesta):** con n = 157 y ~10 % de problemas discordantes entre variantes, el IC95 % de una diferencia pareada es de ±5 pp. **El efecto que publica el paper para MBPP+ (+1,6 pp de APM sobre IM) no es detectable con este test.** Para APM frente a IM hará falta más test: añadir HumanEval+ (164; requiere descargarlo y un adaptador de ejecución, pendiente) o varias semillas de calibración.
- **Contaminación:** Qwen2.5 pudo ver MBPP en preentrenamiento. Afecta a los tres brazos por igual, pero limita la lectura absoluta.

## 6. Prueba técnica pequeña (ejecutada en este Mac, sin descargas)

### 6.1 Perfilado real y recorte con las proporciones de 7B→1.5B
Donante: Qwen2.5-0.5B real. «Receptor»: arquitectura hipotética con las mismas proporciones que 7B→1.5B (canales 896→384, Q 14→6, KV 2→1, MLP 4864→2304). No existe un receptor real con esa forma, así que se mide el coste y el slice **por sí solo** (μ = 1). `configs/hito_a_apm_smoke_selfprune.yaml`.

| Medida | Valor |
|---|---|
| Perfilado (128 ej., 15 602 tokens, bf16, MPS) | **43,7 s**, RSS pico **0,94 GB** |
| Construcción de cada slice | < 1 s |
| Solapamiento IM∩APM | capas 1,0 · canales 0,43 · Q 0,49 · KV 0,54 · MLP 0,48 (≈ lo esperado por azar) |

NLL de código por token en la sonda (64 ej., 3313 tokens):

| Modelo | NLL |
|---|---:|
| 0.5B completo | 1,128 |
| **Slice APM** | **10,39** |
| Slice IM | 11,72 |
| Slices aleatorios (3 semillas) | 11,95 / 12,44 / 12,93 |
| *Uniforme sobre el vocabulario* | *11,93* |

**Lectura:** el orden **APM < IM < aleatorio** es coherente con la premisa del paper: la selección guiada conserva más función que el recorte fijo o el azar. Pero los tres slices están **cerca de la uniforme**: con este nivel de recorte un slice no es un modelo funcional, y cualquier efecto tiene que venir de la inyección con μ pequeño. Son 1 modelo y 64 ejemplos, sin réplica: **no es evidencia de que APM mejore un receptor.**

### 6.2 Circuito completo con un par del mismo tamaño
Donante Qwen2.5-Coder-0.5B → receptor Qwen2.5-0.5B, 16 problemas de dev y 16 de test tomados de la partición de diagnóstico (las particiones reales no se tocan). `configs/hito_a_apm_smoke_harness.yaml`.

- Terminan las fases `profile`, `slice`, `dev`, `lock`, `test` y `report`; `resources` y `slice_nll` se probaron con las otras configuraciones. La congelación de μ funciona, y `test` se niega a arrancar sin `locked_choice.json` (comprobado con la configuración real).
- **Cordura:** con formas iguales, APM e IM eligen todos los índices. El slice resulta **idéntico byte a byte al donante** (sha256 `aff8914e…`, el mismo de `BASELINE_V03_LOCKED.md`) y las generaciones de IM y APM coinciden ejemplo a ejemplo en dev y test.
- Coste: fusión de 1-6 s; generación + tests ≈ 1,2-1,6 s/problema (0.5B, FP32, lotes de 8, 256 tokens); memoria MPS 2,7 GB.
- Las cifras de pass@1 (n = 16; p. ej. R 7/16 frente a fusión 4/16 en dev) **no miden APM** y no deben citarse como resultado.

## 7. Experimento comparativo: no ejecutado

**Motivo:** no cabe en disco (29,8 GB necesarios, 8,8-13,4 GB libres). `scripts/hito_a_apm.py configs/hito_a_apm.yaml resources` lo detecta y sale con código 2. **No se lanzó ninguna descarga.**

**Coste estimado en este Mac, una vez haya disco** (extrapolado del 0.5B, no medido):

| Fase | Estimación |
|---|---|
| Descarga | 18,3 GB |
| Perfilado del 7B (128 ej.) | 5-15 min |
| Slices IM + APM | ~2-5 min |
| Dev: 9 variantes × 333 problemas | 2,5-6 h |
| Test: 3 variantes × 157 problemas | 0,5-1 h |

**Para ejecutarlo:**
```bash
# 1) liberar ≥30 GB y comprobarlo
PYTHONPATH=src python scripts/hito_a_apm.py configs/hito_a_apm.yaml resources
# 2) descargar (revisiones fijadas)
python scripts/download_models.py hito_a
# 3) fases, en orden (todas reanudables)
for f in profile slice slice_nll dev lock test report; do
  PYTHONPATH=src python scripts/hito_a_apm.py configs/hito_a_apm.yaml $f || break; done
# el 7B puede borrarse tras `slice` (solo hace falta para perfilar y recortar)
```
En el PC Windows/Linux, el sandbox pasa a ser Docker (tiene que estar arrancado) y el dispositivo, CUDA. El 1.5B en FP32 (6,2 GB) cabe en los 8 GB de la RTX 3060 Ti con lotes pequeños.

### 7.1 Ejecución en Kaggle (preparada)

Notebook: [`notebooks/kaggle_hito_a_apm.ipynb`](../notebooks/kaggle_hito_a_apm.ipynb). Clona la rama `hito-a-apm`.

1. Settings → Accelerator **GPU P100** (preferible) o **GPU T4 x2**, Internet **On**.
2. **Save Version → Save & Run All**.
3. Al terminar, descargar `hito_a_apm_results.zip` de Output. Si la sesión se corta tras el desarrollo, queda `hito_a_apm_after_dev.zip`.

Qué hace, en orden: tests → `resources` (se detiene si no hay disco, RAM o sandbox) → descarga con revisiones fijadas → `profile` → `slice` → `slice_nll` → borra la copia del 7B → `dev` → `lock` → `test` → `report`. Con 2 GPUs, `dev` y `test` reparten las variantes en un proceso por GPU (`--only`). La sandbox es `MM_SANDBOX=subprocess`, porque Kaggle no tiene Docker. Cualquier fallo detiene el notebook.

**Estimación (no medida): 1,5–3,5 h**, dentro del límite de sesión de 12 h. El cuello de botella es la generación del 1.5B en FP32, que está limitada por ancho de banda de memoria. Por eso P100 (~730 GB/s) debería ir unas 2× más rápido que una T4 (~320 GB/s); con T4 x2 se compensa repartiendo las variantes.

**Cambios en `configs/hito_a_apm.yaml` hechos antes de cualquier ejecución real** (ninguno depende de resultados):

| Parámetro | Antes | Ahora | Motivo |
|---|---|---|---|
| `profile_dtype` | bfloat16 | float32 | T4 y P100 no tienen bf16 nativo; fp16 puede desbordar activaciones |
| `batch_size` | 8 | 16 | GPU de 16 GB |
| `min_ram_gb` | 16 | 12 | Pico real estimado: ~5 GB |

**Pendiente de verificar en la propia ejecución:** el disco libre real de Kaggle en `/tmp` (las fuentes no coinciden). La fase `resources` lo mide antes de descargar.

## 8. Recomendaciones para el siguiente paso

1. **Ejecutar §7** cuando haya disco; es la única pieza que falta en el Hito A.
2. **Añadir el brazo RAND** (≥3 semillas, ya implementado) **antes** de ver el test. Sin él, una mejora de APM podría deberse a «cualquier inyección pequeña» y no a la selección guiada, dado que las coordenadas de 7B y 1.5B no están alineadas. Hacerlo implica un pre-registro nuevo de la configuración.
3. **Aumentar la potencia del test:** adaptador de HumanEval+ y, si se puede, varias muestras de calibración.
4. No avanzar a Theseus ni a cirugía con Claude hasta tener la tabla R / IM / APM en test.

## 9. Reproducibilidad

| Artefacto | sha256 (16) |
|---|---|
| `configs/hito_a_apm.yaml` | `b538bbd75347f152` (tras los cambios de §7.1) |
| `configs/hito_a_apm_smoke_selfprune.yaml` | `ba8dfdc589f75d6d` |
| `configs/hito_a_apm_smoke_harness.yaml` | `1419ea9e2198903a` |
| Estadísticas, prueba 6.1 | `1217faac72092b1d` |
| Estadísticas, prueba 6.2 | `a8afc00ddcf12990` |
| Slice APM / IM, prueba 6.1 | `1e60797749b41e4e` / `0e045c504323ea75` |
| `locked_choice.json`, prueba 6.2 | `6fc3b2ab32ae9403` |

- **Modelos:** Qwen/Qwen2.5-7B@`d149729398750b98c0af14eb82c78cfe92750796` (Apache-2.0); Qwen/Qwen2.5-1.5B@`8faed761d45a263340a0528343f099c05c9a4323` (Apache-2.0). Configuraciones reales en `configs/arch/`.
- **Resultados por ejemplo:** `results/experiments/hito_a_apm_smoke/*/{dev,test}/{pass,nll}_*.jsonl`. Cada `meta_*.json` registra receta, hash del slice, tiempos, memoria, dtypes, versiones y commit.
- Los pesos de los slices de prueba se borraron para liberar disco (son regenerables y sus hashes están en los `manifest.json`). `.gitignore` excluye `results/experiments/**/slices/**/model.safetensors`.
- **Ningún checkpoint original se modificó.** `build_slice` escribe en directorios nuevos, se niega a sobrescribir, y hay un test que compara los hashes de los originales antes y después.

## Archivos nuevos o modificados

- Nuevos: `notebooks/kaggle_hito_a_apm.ipynb`, `src/model_maxxing/transfer/apm.py`, `src/model_maxxing/transfer/apm_profile.py`, `scripts/hito_a_apm.py`, `tests/test_apm.py`, `configs/hito_a_apm*.yaml`, `configs/arch/*.config.json`, `results/experiments/hito_a_apm_smoke/`.
- Modificados: `tests/test_sandbox_subprocess.py` y `evaluation/runner.py` (guarda de Linux), `scripts/download_models.py` (conjunto `hito_a`), `.gitignore`.
