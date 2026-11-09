# Model Maxxing (prev) — estado y plan de ejecución

Escrito para: Rubén, para continuar el experimento en el PC de casa (GPU de 8 GB).

## 1. Lo que ya tenemos

**Infraestructura (probada, 18 tests con tensores sintéticos):**
- Parches de pesos tipados (bloque o rango bajo): validación (lista de tensores, norma, finitud, `||P||>0`), aplicación sin mutar originales, hash, guardado y rollback.
- Transferencia: interpolación y task vector; medida de supervivencia del parche tras la transferencia.
- Evaluación: MBPP (ejecución aislada), MedMCQA 3-shot, NLL, bootstrap pareado.
- Herramientas de diagnóstico para el cirujano (`agents/diagnostics.py`): activaciones, contraste de canales código/medicina, direcciones singulares, aciertos vs fallos, historial, sensibilidad a parches.
- Constructores de brazos con presupuesto idéntico (`surgery/arms.py`).

**Resultados (todos con 0,5B; muestras pequeñas, orientativos):**

| Hallazgo | Dato |
|---|---|
| Barrera de interpolación Base↔Coder | **Real, no un bug.** NLL de código 1,39 → 2,44 (α=0,5) → 1,30. Embeddings con diferencia relativa 1,02. Ver `docs/BARRIER_REPORT.md`. |
| Cirugía v0.2 (bloque 16×16) | Parche aplicado, persistente y reversible, pero **efecto nulo** (A=C=D=44/100). |
| Par con procedencia (v0.3) | R = Med-Pre-Trained-92k (CPT PubMed), D = codeforces-SFT, ambos de Qwen2.5-0.5B. |
| Complementariedad del par | **Débil.** MedMCQA (300): B 0,340 · R 0,313 · D 0,333, R no supera a nadie. MBPP (100): B 47 · R 39 · D 48. NLL PubMed: R 2,270 < B 2,299 < D 2,329 (la especialidad de R es real pero pequeña). |
| Tamaño de los deltas | ‖Δ_D‖=1,77 frente a ‖Δ_R‖=14,0, casi ortogonales (cos 0,005). |
| **Confusor importante** | Interpolar R→D mejora a ambos extremos, pero **interpolar R→B también** (bio 2,250, código 1,437). La mejora es encoger el sobreajuste del receptor, no el donante. |
| Sensibilidad por capa | Efectos de ≤0,003 nats casi en todas partes. Deshacer parte de Δ_R (`dR-`) ayuda a código y a bio a la vez en capas 0–5 y 12–19. Las capas 2, 21 y 23 con la dirección del donante empeoran (hasta +0,052). |

**Lectura honesta:** de momento no hay evidencia de que Claude aporte una modificación útil. Tampoco se ha probado lo contrario: el experimento decisivo no se ha ejecutado.

## 2. Lo que falta

1. **`scripts/run_arms.py`** (en construcción): genera y evalúa los brazos con el presupuesto pre-registrado (N0=1,0; ≤12 tensores `down_proj`; rango ≤8):
   - A sin cirugía · B SVD de Δ_D · B2 SVD de −Δ_R · C regla fija sobre la tabla de sensibilidad · **D diseñado por Claude** · E control aleatorio igualado a D.
   - Cada parche evaluado solo y con la transferencia `R* + 0,5·(D−B)`.
   - Referencia extra: R→B (α=0,5), el control del confusor.
2. **El diseño del brazo D** debe quedar fijado en `configs/arm_D_claude_design.json` y commiteado **antes** de evaluar.
3. **Informe final** con intervalos de confianza y decisión Go/Pivot/Stop (plantilla en `docs/SPIKE_REPORT.md`).

**Criterio pre-registrado:** un candidato es válido si MedMCQA ≥ R − 5 pp y NLL biomédico en test ≤ R + 0,02. La mejor ganancia en MBPP se toma solo entre los válidos. Datos de evaluación separados de los visibles para el agente.

## 3. Cómo ejecutarlo en casa
```bash
git clone <tu repo> && cd modelmaxxing-prev
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[eval,dev]"                              # instala torch con CUDA si tu pip lo detecta
python scripts/download_models.py                         # ~4 GB, revisiones fijadas
pytest -q                                                 # debe dar 18 passed
PYTHONPATH=src python scripts/run_arms.py                 # reanudable: salta lo ya calculado
```
- **Sandbox:** el runner usa `sandbox-exec` en macOS; en Linux/Windows usa **Docker** (sin red, 512 MB, 1 CPU). Necesitas Docker arrancado. Si no hay ninguno de los dos, se niega a ejecutar código.
- El dispositivo se elige solo (`cuda` > `mps` > `cpu`).

## 4. ¿Tardará mucho en 8 GB de VRAM?

**No debería.** Estimación (extrapolada de tiempos medidos en el Mac M4; no he medido en CUDA):

| Parte | Tiempo estimado |
|---|---|
| Generación MBPP, 100 problemas | ~2–2,5 min por configuración (medido: 1,1–1,5 s/problema en el Mac) |
| Ejecución de tests (Docker) | ~1–2 min por configuración (~0,5–1 s por arranque de contenedor) |
| MedMCQA 300 + NLL bio | ~1 min por configuración |
| Construir parches (SVD en CPU) y cargar pesos | segundos por variante |
| **13 configuraciones en total** | **~1 h** (rango razonable: 45 min a 1,5 h) |

- **Memoria:** el modelo en fp16 ocupa ~1 GB, así que 8 GB de VRAM sobran, incluso en fp32 (~2 GB). Lo que sí importa es la **RAM**: se cargan B, R y D (~3 GB) más deltas en fp32 (~2 GB), así que conviene tener ≥8 GB libres.
- **Lo que no hace falta repetir:** la tabla de sensibilidad (~25 min en el Mac) ya está en `results/metrics/`.
- La descarga de ~4 GB es lo que más depende de tu conexión.

## 5. Riesgos
- Con ≤0,003 nats por capa, lo esperable es un **resultado nulo** o diferencias dentro del ruido (MBPP con 100 problemas tiene un IC de ±9 pp). Un resultado nulo con esta muestra no descarta cirugías mayores.
- Si el resultado importa para el TFG, conviene repetir con más problemas (MBPP test completo, 257) y con varias semillas del control aleatorio. Con la GPU de casa sería ~2,5× el tiempo.
- La licencia del donante (`codeforces-SFT`) no está declarada: revisarla antes de publicar nada derivado.
