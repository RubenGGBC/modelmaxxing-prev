# Informe: barrera de interpolación (v0.3, Prioridad 1)
Reproducible con `PYTHONPATH=src python scripts/diagnose_barrier.py` → `results/reports/barrier_report.json`.
Par: Qwen2.5-0.5B (R sustituto) ↔ Qwen2.5-Coder-0.5B (D). Corpus fijo: código = MBPP test[100:160] (5 chunks, 2560 tokens, fuera de los 100 de evaluación); texto = WikiText-2 test (24 chunks, 12288 tokens). Muestra de código pequeña: orientativa.

## Veredicto
La caída es una **barrera de pérdida real**, no un problema de implementación.

## Evidencia
1. **Matemática:** α=0 y α=1 reproducen R y D bit a bit; dos formulaciones independientes de la interpolación coinciden (error FP32 ≤1.5e-5); sin NaN/Inf.
2. **Dtype:** los checkpoints son **bf16** (no fp16). El pipeline guarda en bf16 (dtype de R) y se carga en fp16 para evaluar. El error de redondeo máximo (0.5) está en biases de k_proj de magnitud grande (error relativo ~2^-8), no en pesos. FP32 vs fp16 en NLL difiere ≤0.03: no explica la caída. Nota: informes anteriores decían "fp16" para el almacenamiento; era bf16.
3. **Pesos compartidos:** ambos con `tie_word_embeddings=True` y sin `lm_head` en el checkpoint; al cargar solo falta `lm_head.weight` (se ata a embed_tokens), sin claves inesperadas. NLL en α=0 y α=1 es coherente (código 1.386 / 1.305), así que el atado funciona.
4. **Curva NLL (FP32), código | texto:** α=0: 1.386|2.852; 0.1: 1.471|2.934; 0.25: 1.744|3.533; 0.5: **2.443|4.516**; 0.75: 1.780|3.525; 0.9: 1.418|3.089; α=1: 1.305|2.991. Sobre la cuerda lineal entre extremos, el exceso en α=0.5 es +1.10 (código) y +1.59 (texto) nats/token. Forma de joroba, simétrica-ish: típica de barrera.
5. **Localización (α=0.5 solo en un grupo, código|texto):** embed 2.36|4.87; attn 1.95|3.75; mlp **3.17|5.40**; norms 1.47|2.92; capas 0–11 2.51|4.00; capas 12–23 2.02|3.69. No hay un único culpable: mezclar solo los MLP es peor que mezclar todo (interacciones no aditivas); los embeddings solos ya rompen el modelo.
6. **Embeddings:** diferencia relativa ‖E_D−E_R‖/‖E_R‖ = **1.02**. Son tan distintos como la propia matriz, lo que apunta a que los espacios de representación no están alineados. Es coherente con la advertencia de v3 de no asumir una trayectoria de entrenamiento común; no es prueba de procedencia.

## Implicaciones
- La interpolación lineal y el task vector entre estos dos checkpoints no son operadores viables para este par (α útiles solo cerca de los extremos: α=0.1 ya cuesta +0.09 nats en código).
- Un parche local pequeño no puede esperarse que cambie esto; la cirugía tendría que atacar la desalineación (p. ej. alinear espacios/permutaciones antes de mezclar) o usar un par con procedencia verificable.
- Limitaciones: 5 chunks de código; un solo par; receptor sustituto = base.
