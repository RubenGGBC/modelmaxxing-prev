# Hito B — Theseus: traslado de task vectors 7B → 1.5B (diseño)

**Estado:** diseño aprobado (2026-11-10): τ = Coder-7B − Base-7B; test = MBPP final + HumanEval+. **Parte de:** Hito A = PIVOT (APM: +1,9 pp de pass@1 no significativo; −1 % de NLL significativo). Ver [HITO_A_APM_REPORT.md](HITO_A_APM_REPORT.md).

## 1. Método (Theseus, arXiv 2602.12952) y adaptación a Qwen2

**Idea.** No inyectar pesos del modelo grande, sino el **cambio que produjo su ajuste** (task vector), expresado en las coordenadas del modelo pequeño:

- τ_A = θ(Qwen2.5-Coder-7B) − θ(Qwen2.5-7B)   (misma arquitectura, revisión `0396a761…`, Apache-2.0)
- Por cada capa lineal ℓ, se pasan los **mismos tokens** por A = Qwen2.5-7B y B = Qwen2.5-1.5B y se acumulan las covarianzas cruzadas de entradas y salidas:
  C_in = Σ H_in,Aᵀ H_in,B,  C_out = Σ H_out,Aᵀ H_out,B
- Procrustes (sin entrenamiento): C = UΣVᵀ → T = UVᵀ (semiortogonal, d_A × d_B).
- Traslado: **τ_B = T_outᵀ · τ_A · T_in** (forma d_out,B × d_in,B), y receptor final **θ_B + α·τ_B**.

  Comprobación de formas: si x_B ≈ x_A·T_in, entonces Δy_B ≈ x_B·T_inᵀ·τ_Aᵀ·T_out, de donde τ_B = T_outᵀ τ_A T_in. La Eq. 10 del paper tiene las transposiciones inconsistentes; esta es la forma coherente.

**Ventajas en nuestro caso:**
- **Sin desajuste de longitud de secuencia:** A y B comparten tokenizer, así que los tokens se emparejan uno a uno y no hace falta la interpolación que usa el paper en visión.
- **Sin desajuste de profundidad:** 28 = 28 capas, emparejadas por índice.

**Decisiones nuestras donde el paper no especifica** (se documentarán en el código):

| Pieza | Decisión |
|---|---|
| Capas trasladadas | q, k, v, o, gate, up, down (+ sesgos q/k/v: τ_b,B = τ_b,A·T_out) |
| Embeddings, lm_head, RMSNorm | **No se trasladan** en la variante principal; el paper no lo trata |
| Entrada común de q/k/v y de gate/up | Una sola T_in por capa y tipo de entrada (comparten H_in) |
| **RoPE en q/k** | Una rotación general del espacio de salida de q/k mezcla los pares de frecuencias de RoPE y no preserva la atención. Variante **TH-noQK**: no traslada q/k. Variante fiel **TH**: traslada todo |
| Datos de alineamiento | MBPP `diagnostic` (prompt + solución de referencia), ~256 ejemplos ≈ 30 000 tokens; hacen falta más tokens que la dimensión mayor (8960) |
| α | Rejilla en dev (§3). El delta Coder es grande, así que α = 1 probablemente rompe el modelo |

### 1.1 Hallazgos al implementar (cambios respecto al diseño inicial)

1. **Salida de gate/up: alinearla por separado no está identificado.** La salida de un Linear que expande tiene rango ≤ d_in: en el 7B, la salida de gate tiene 18 944 dimensiones pero rango ≤ 3584, y su covarianza cruzada con el 1.5B, rango ≤ 1536 de 8960. Procrustes elegiría al azar más del 80 % de la rotación. **Decisión:** gate.out, up.out y down.in indexan el mismo espacio de neuronas, así que se alinea **una vez** con la post-activación (entrada de down_proj), que tiene rango completo. Comprobado en un test con modelo diminuto: covarianza de salida de gate con cond ≈ 1e9 frente a ≈ 70 en down.in. Además ahorra 2 de las 3 covarianzas grandes.
2. **Precisión numérica: float64 obligatorio.** Las covarianzas sin centrar de Qwen tienen condicionamiento de **1e9 a 1e12**, por los sesgos enormes de q/k y las activaciones masivas. En el 0.5B real, con float32 la rotación del caso A = B salía con error de hasta 1,1 en q_proj y 0,71 en down_in; con float64 baja a ≤ 1e-7. La acumulación y la SVD se hacen en float64 (en una T4 es más lento: ver §4).
3. **Direcciones no identificables.** Donde B no tiene energía en los datos (p. ej. W_q de la capa 0 del 0.5B tiene rango 879/896), la rotación no está determinada y el delta se proyecta de forma arbitraria. Es una propiedad del método. Se registra por grupo (`null_dims`) en `alignment_diagnostics.json`.
4. **Anchuras distintas.** T = UVᵀ maximiza la correlación tr(TᵀC). Solo coincide con el mínimo de ‖H_A·T − H_B‖ si las activaciones de A son isotrópicas, y las de Qwen no lo son. Es el método del paper y lo dejamos así, con la salvedad registrada.
5. **HumanEval+:** HumanEval/32 se excluye porque su solución canónica falla con el test de la revisión fijada. Las otras 163 pasan en nuestro sandbox (máx. 5,7 s) y un cuerpo `pass` no aprueba ninguna.

## 2. Riesgo principal (antes de gastar GPU)

τ_A = Coder − Base **no es un ajuste fino pequeño**: es preentrenamiento continuado con 5,5 billones de tokens. En el par 1.5B, ‖Coder − Base‖ es el **63 %** de ‖Base‖ (`results/experiments/maxxing_v04/selection/weights.json`), y en v0.3 la interpolación Base ↔ Coder (0.5B) mostró una **barrera**: el NLL pasó de 1,39 a 2,44 con α = 0,5. Theseus supone un delta pequeño y lineal, así que puede no funcionar con este τ. Por eso la rejilla de α empieza en valores pequeños y hay un control con un τ aleatorio.

## 3. Brazos y evaluación (a pre-registrar en `configs/hito_b_theseus.yaml`)

| Brazo | Qué es | Para qué |
|---|---|---|
| R | Qwen2.5-1.5B | Referencia |
| **TH** | Theseus completo | Método fiel |
| **TH-noQK** | Theseus sin q/k | Adaptación por RoPE |
| NAIVE | τ_A truncado a la forma de B, sin alinear | El «IM» de Theseus: aísla el efecto del alineamiento |
| RANDτ | τ aleatorio con la misma norma por tensor que τ_A, por las **mismas** T | Control del propio paper: ¿importa el contenido de τ o basta una perturbación así? |
| APM μ=0,02 | Del Hito A | Comparar con la línea base anterior |

Controles **exploratorios** del Hito A, que aprovechan la misma ejecución: «solo encoger» θ = 0,98·θ_B, y slice RAND de APM con μ = 0,02.

**Elección de α en dev (333 problemas), en dos etapas fijadas antes:**
1. NLL de código para α ∈ {0,05; 0,1; 0,25; 0,5; 1,0} en cada brazo (barato, ~80 s por variante).
2. pass@1 solo para los **2 α con mejor NLL** de cada brazo.
3. Se elige por pass@1 (empate → NLL → α menor) y se congela con `lock` antes del test.

**Test:** MBPP `final` (157), ya usado una vez en el Hito A con brazos congelados de antemano, **más HumanEval+ (164), nuevo**, que necesita un adaptador. Métrica primaria: pass@1 combinado (321). Criterio: +3 pp frente a R y frente a APM, con IC95 > 0.

## 4. Coste estimado (Kaggle T4 x2, no medido)

| Fase | Estimación |
|---|---|
| Descarga 7B + Coder-7B + 1.5B | ~34 GB, ~3 min |
| Alineamiento (A y B capa a capa a la vez, covarianzas en GPU, **float64**) | Mayor matriz: 18 944 × 8960 en float64 ≈ 1,4 GB (solo down.in, §1.1). 28 SVD grandes en float64 en T4 (~1/32 de la velocidad de FP32). **~45-90 min, el paso más incierto.** En el Mac (0.5B, CPU float64): ~20 s por capa |
| τ_B por brazo | Segundos-minutos (en streaming) |
| NLL dev (4 brazos × 5 α) | ~30 min |
| pass@1 dev (8 + R) | ~40 min en 2 GPUs |
| Test (321 × ~5 brazos) | ~40 min |
| **Total** | **~3-4 h** (dentro de la sesión de 12 h) |

## 5. Orden de trabajo

1. **Implementación + tests locales sin descargas:**
   - **Rotación plantada:** B = A con el residual rotado por una ortogonal Q, y neuronas MLP y cabezas permutadas. Theseus debe recuperar el delta exacto y B + τ_B debe igualar funcionalmente a A + τ_A rotado.
   - **Auto-traslado real (0.5B):** A = B = Qwen2.5-0.5B y τ = Coder-0.5B − 0.5B. Con T = I, debe reproducir el Coder-0.5B.
   - **Formas** con modelos sintéticos de anchuras distintas, incluido el caso con embeddings atados.
2. Adaptador HumanEval+ con tests.
3. Pre-registro y notebook de Kaggle; después, ejecución.
