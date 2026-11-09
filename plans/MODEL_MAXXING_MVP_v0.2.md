# MODEL MAXXING — MVP v0.2
## Especificación corregida para Claude: cirugía previa de pesos + transferencia

**Estado:** Propuesta de implementación. Ningún resultado experimental se presupone.
**Contexto:** TFG de Ingeniería del Software.
**Prioridad:** demostrar una intervención de pesos diseñada por el LLM frontera, no limitarse a un planificador de recetas de merging.
**Alcance inmediato:** spike de viabilidad de 48 horas de calendario; después, MVP ampliable.

---

## 0. Corrección de rumbo obligatoria

El documento v0.1 definía un laboratorio de interpolación y task vectors y posponía el LLM frontera. **Eso NO prueba la hipótesis central de Model Maxxing.** Los baselines se conservan, pero la primera versión demostrable debe ejecutar al menos una **cirugía previa de pesos propuesta por Claude**, seguida de un algoritmo de transferencia y evaluación.

**No implementar solamente:** un chatbot que elija `TIES`, `DARE`, `alpha`, capas o pipelines. Esa funcionalidad puede ser un control experimental, pero no constituye el cirujano.

**Modelo conceptual:**

```text
Receptor R + Donante D (+ Base B si la procedencia lo justifica)
                         |
           Diagnóstico de pesos, activaciones y errores
                         |
                CLAUDE — WEIGHT SURGEON
          inspecciona y CONSTRUYE una edición P
                         |
                  Receptor R* = R + P
                         |
       Transferencia convencional (MergeKit u operador propio)
                         |
             Evaluación de capacidades protegidas
                         |
        Checkpoint final, parche, receta, reporte y rollback
```

El parche **P** debe modificar numéricamente al menos un tensor del checkpoint antes de invocar la transferencia. La herramienta que materializa la edición es software determinista; el contenido, estructura o dirección de la intervención es propuesto por Claude a partir de observaciones reales. Sin parche real aplicado, el experimento debe etiquetarse **PLANNER, no SURGEON**.

## 1. Pregunta científica central

> ¿Puede un LLM frontera, usando información del receptor y del donante, construir una edición previa de pesos que mejore una transferencia posterior de capacidades frente a la misma técnica sin cirugía y frente a controles numéricos con igual presupuesto?

**Hipótesis separadas:**
- H1. El donante supera al receptor en la habilidad objetivo y el receptor destaca en una habilidad protegida.
- H2. Una transferencia convencional produce un compromiso medible o interferencias.
- H3. Claude puede producir un **parche no trivial y reproducible** sobre los pesos a partir de diagnósticos.
- H4. `Transfer(R + P_Claude, D)` mejora el compromiso respecto a `Transfer(R, D)` y a parches de control, con evaluación fuera de los ejemplos de optimización.

H3 se puede demostrar técnicamente sin que H4 sea cierta. **Si H4 falla, informar del resultado negativo.**

## 2. Modelos y compatibilidad

**Primera opción de trabajo (no garantizada):**
- Base B: `Qwen/Qwen2.5-3B`.
- Receptor R: checkpoint local Qwen2.5-3B especializado en medicina, si existe y se comprueba su procedencia.
- Donante D: `Qwen/Qwen2.5-Coder-3B`.
- Habilidad objetivo: programación Python.
- Capacidad protegida: conocimiento médico.

Comprobar arquitectura, correspondencia de tensores, tokenizer, revisión exacta, licencias, familia Base/Instruct, historial de fine-tuning, y que el modelo receptor sea verdaderamente especializado. No interpretar `base_model` como prueba suficiente de que los parámetros puedan restarse funcionalmente.

Si no se puede demostrar parentesco de B y D, **no llamar task vector validado** a `D-B`; se puede investigar una operación compatible marcada exploratoria, pero no apoyar conclusiones fuertes en ella.

Un 7B → 3B **no** entra en este spike: necesitaría poda, transporte, alineamiento o destilación; MergeKit estándar no convierte automáticamente las dimensiones.

## 3. Qué significa que Claude «modifica pesos»

El agente NO lee ni escribe miles de millones de números. El sistema ofrece herramientas para inspeccionar regiones seleccionadas y construir parches compactos. Dos modos implementables:

### Modo 1 — Cirugía de bloque pequeño (prueba de autenticidad)
1. El profiler selecciona un módulo de atención o MLP compatible y una submatriz acotada, p. ej. 16×16 o 32×32.
2. Claude recibe valores numéricos de dicha región, estadísticas de `D`, `R` y `B`, y ejemplos de error.
3. Claude **emite una matriz de corrección numérica propia** o factores cortos `u`, `v` para `P_block = u vᵀ`, sujetos a límites de norma.
4. El motor aplica la edición al receptor; calcula el hash del parche y registra la matriz modificada.
5. Se verifica que `||P|| > 0`, que no hay NaN/Inf y que el cambio **permanece** tras la transferencia.

Este modo demuestra intervención directa, **pero un bloque minúsculo puede no producir mejoras observables en un 3B**. Un resultado nulo no debe falsamente descartarse como prueba de que la aproximación nunca funcionará.

### Modo 2 — Cirugía compacta guiada por señales (MVP recomendado)
1. Las herramientas extraen un conjunto pequeño de **direcciones numéricas fundamentadas**, por ejemplo diferencias de pesos, direcciones SVD de activaciones, estimaciones de sensibilidad o componentes de deltas.
2. Claude inspecciona esas direcciones y construye un parche propio como combinación y transformación acotada de ellas:

   `P = Σ_i c_i * B_i`, con posibles proyecciones, reescalados y recortes por subespacio.

3. Claude **especifica coeficientes con signo, transformaciones y componentes del parche**, y no solo la receta MergeKit; un motor ejecuta la transformación exacta.
4. El sistema guarda `P` y una receta que recrea exactamente los tensores.

**Limitación metodológica:** si Claude solo elige `c_i`, podría ser indistinguible de una búsqueda convencional en pocos hiperparámetros. Por ello conservar el modo 1 como prueba de intervención numérica y comparar el modo 2 con un optimizador que use **la misma base `B_i`**.

### Restricciones de seguridad del parche
- Lista permitida de tensores/capas y dimensiones.
- `max_relative_norm` por bloque, norm clipping y reversión.
- Salidas estructuradas validadas: tipos, tamaños, valores finitos.
- Los checkpoints originales no se mutan.
- Se prohíbe código arbitrario suministrado por el LLM para escribir tensores; usar operaciones tipadas.
- Al crear un candidato, confirmar numéricamente `R* != R` y registrar los cambios.

## 4. Secuencia experimental imprescindible

Todas las configuraciones deben usar **el mismo par de modelos**, los mismos conjuntos de validación y un presupuesto de evaluaciones comparable.

| ID | Variante | ¿Claude decide receta? | ¿Claude construye P? | Objetivo de control |
|---|---|---|---|---|
| A | `Transfer(R,D)` con MergeKit u operador equivalente | No | No | Baseline de transferencia |
| B | `Transfer_planner(R,D)` | Sí | No | Efecto de planificación |
| C | `Transfer(R + P_Claude,D)` | Puede | **Sí** | Hipótesis central de cirugía previa |
| D | `Transfer(R + P_control,D)` | No | No | Control numérico igualado en norma, región, rango y presupuesto |
| E | `R + P_Claude` sin transferencia | No | **Sí** | ¿La mejora procede solo del parche? |

**Añadir también controles fijos:** R, D, B (si se utiliza), interpolación `alpha=0/1` y transferencia sin preacondicionamiento. Si los recursos son limitados, ejecutar A, C, D y E primero; B es muy útil como ablación posterior.

Los parches de control deben ser, como mínimo: (i) misma norma y estructura con dirección aleatoria, (ii) misma familia de direcciones con elección numérica convencional (aleatoria o heurística). Si el presupuesto lo permite, utilizar una búsqueda optimizadora como baseline adicional.

**Cuidado matemático:** con interpolación directa:

`Transfer(R + P,D) = (1-alpha)*(R + P) + alpha*D`

La contribución efectiva del parche es `(1-alpha)*P` y **desaparece en alpha=1**. Por tanto, no interpretar como fallo de cirugía una prueba donde el algoritmo cancela su efecto. Probar coeficientes `alpha<1`, registrar `||W_final(P)-W_final(0)||`, o preferir una transferencia aditiva bien justificada donde P sobreviva.

Para deltas verificados:

`Delta_D = W_D - W_B`

`R* = R + P_Claude`

`W_final = R* + alpha * Delta_D`

Con TIES/DARE, verificar el tratamiento exacto de los deltas y si sus operadores preservan, alteran o anulan la edición anterior.

## 5. Herramientas del agente — contrato mínimo

Implementar interfaces tipadas, sin dar al modelo acceso irrestricto al entorno:

```python
inspect_model(model_id) -> ModelManifest
compare_performance(receiver_id, donor_id, split) -> GapReport
inspect_weight_region(model_id, tensor_name, slice_spec) -> NumericRegion
inspect_delta_region(base_id, donor_id, tensor_name, slice_spec) -> DeltaRegion
inspect_activation_summary(model_id, task_ids, module_names) -> ActivationReport
propose_weight_patch(candidate_parent_id, patch_spec) -> PatchId
validate_patch(patch_id) -> PatchValidation
apply_weight_patch(model_id, patch_id) -> CandidateId
run_transfer(receiver_candidate_id, donor_id, transfer_spec) -> CandidateId
evaluate(candidate_id, benchmark_split) -> EvaluationReport
compare_candidates(candidate_ids) -> ComparisonReport
rollback(candidate_id) -> ModelId
```

`patch_spec` debe permitir al menos una edición numérica de rango bajo o una corrección pequeña explícita, además de las transformaciones basadas en señales.

**Guardar**: modelo y revisión, tensor y slices, valores o base de construcción de P, justificación de Claude, llamada a la herramienta, hash del parche, norma relativa, prueba de persistencia, algoritmo posterior, seeds, resultados y coste.

## 6. Datos y evaluación

- Python: MBPP (partición controlada), reservar HumanEval y un subconjunto oculto para el informe final. El conjunto de desarrollo no puede reutilizarse como evaluación final.
- Medicina: conjunto de preguntas con respuestas correctas, apropiado para el checkpoint médico y separado de sus datos de ajuste si es posible.
- Primero medir **complementariedad**: `Python(D)>Python(R)` y `Medicine(R)>Medicine(D)` con diferencias relevantes para el tamaño de muestra.
- Usar prompts y decodificación idénticos, resultados por ítem, cambios correcto→incorrecto y viceversa, confianza estadística pareada en evaluación final.
- La tolerancia de degradación, inicialmente `5 pp`, se registra antes de optimizar y no se cambia tras inspeccionar resultados.
- Los programas generados se ejecutan en sandbox aislada con límites de tiempo, CPU, memoria y red.
- La conclusión de un spike sobre 20–50 ejemplos es **exploratoria**, no demostración estadística.

## 7. Spike de 48 horas: orden CORREGIDO

### Puerta 0 — Preparación (primeras horas)
- Confirmar modelos, recursos, benchmarks y compatibilidad.
- Ejecutar R y D para confirmar brecha funcional; si el donante domina todo, cambiar receptor.
- Preparar evaluación rápida sobre un subconjunto de diagnóstico y otro de selección.

### Puerta 1 — Transferencia convencional (día 1)
- Implementar `Transfer(R,D)` con MergeKit cuando resulte compatible o con operación propia controlada.
- Comparar `alpha=0`, un valor intermedio y `alpha=1`, y un task vector si hay base verificada.
- Registrar regresiones. No invertir el spike completo en barridos de coeficientes.

### Puerta 2 — **Claude Weight Surgeon** (inicio del día 2)
- Exponer a Claude un par de módulos o regiones concretas y resultados diagnósticos.
- Solicitar 1–3 propuestas de parche **numérico no nulo**; limitar norma, rango y región.
- Generar físicamente `R*`, guardar el parche como artefacto y evaluar `R*` sin transferencia.
- Aplicar sobre `R*` la **misma transferencia** que el baseline A.
- Comprobar que el parche sobreviva a la transformación.

### Puerta 3 — Comparación justa (día 2)
- Repetir con 1–3 parches de control igualados en norma y espacio de búsqueda.
- Comparar A, C, D y E con el mismo presupuesto de evaluaciones; registrar gasto del LLM frontera.
- Producir un informe con efectos medidos y casos de error. No atribuir causalidad a una capa solo por correlación.

### Si no hay tiempo
Mantener el entregable más pequeño que demuestre la idea: **una propuesta de edición numérica por Claude, un parche aplicado y verificable, una transferencia posterior y la comparación con un control de igual tamaño**. Reducir número de modelos o candidatos antes de eliminar la cirugía.

## 8. Criterios de éxito y decisión

**GO técnico** si:
1. Existe un par de modelos con complementariedad funcional.
2. Se implementó un operador de transferencia válido.
3. Claude generó una propuesta de parche propio, se aplicó, persistió tras el operador y se pudo revertir.
4. Los resultados y el experimento son reproducibles.

**GO de investigación (más exigente)** si, además, C presenta una mejora reproducible frente a A y D sin superar la pérdida protegida; esto requiere repetición y un conjunto de evaluación suficiente, no solo el spike.

**PIVOT** si hay transferencia convencional útil pero la cirugía no produce mejoras: investigar regiones mayores, señales mejores, modos de parche alternativos o intervenciones sobre el donante, manteniendo los mismos controles.

**STOP para la hipótesis concreta** si los parches no sobreviven, son incompatibles, o solo funcionan a costa de regresiones intolerables y no hay ruta factible de mejora.

**Nunca:** presentar "Claude modificó pesos" como "Claude mejoró el modelo" sin resultados.

## 9. Entregables obligatorios al agente de programación

```text
model-maxxing/
  configs/
    models.yaml
    spike.yaml
  src/model_maxxing/
    models/compatibility.py
    profiling/weight_analysis.py
    surgery/patch_schema.py
    surgery/patch_builder.py
    surgery/patch_validator.py
    surgery/patch_store.py
    transfer/engine.py
    evaluation/runner.py
    agents/frontier_surgeon.py
    experiments/runner.py
  tests/
    test_patch_is_nonzero.py
    test_patch_norm_bounds.py
    test_patch_persists_after_transfer.py
    test_patch_rollback.py
    test_transfer_baselines.py
  results/
    manifests/
    patches/
    evaluations/
  docs/
    SPIKE_REPORT.md
```

Cada propuesta `patch` debe tener manifiesto y artefacto numérico verificable. El informe debe incluir las diferencias exactas entre C y A y entre C y D, junto a tiempo, VRAM, coste del LLM, errores y limitaciones.

## 10. Reglas de implementación para Claude Code

1. **No sustituir el cirujano por un prompt que solo selecciona TIES/DARE.** Eso sería un planificador.
2. **No reimplementar innecesariamente MergeKit** si una operación segura y compatible ya existe. Encapsular mediante adaptador.
3. Nunca tocar originales ni usar código arbitrario del agente para modificar pesos.
4. Las modificaciones deben ser numéricas, auditables, limitadas y reversibles.
5. Verificar que `R*` difiere de `R` y que el resultado final difiere del baseline sin cirugía en los tensores previstos.
6. Controlar coste de inferencia, espacio en disco y memoria; no cargar todo simultáneamente.
7. Si faltan checkpoints reales, no inventarlos ni simular una mejora. Implementar primero el pipeline con tensores sintéticos para tests y marcar el experimento real como bloqueado.
8. Registrar versiones, hashes, splits y semillas; aislar ejecución de código de benchmarks.
9. Ante un resultado negativo, conservar los datos y explicar hipótesis alternativas.
10. Priorizar un prototipo vertical demostrable sobre una plataforma extensa con interfaz.

## 11. Definición de terminado

El prototipo se considera mínimamente completo cuando puede demostrar, **sobre un modelo compatible y con datos reales**:

- Carga del receptor y donante.
- Diagnóstico de diferencias.
- Petición estructurada a Claude para sintetizar una intervención sobre pesos.
- Aplicación física y validación de `P_Claude`.
- Transferencia posterior con operador existente.
- Evaluación de `R`, `Transfer(R,D)`, `R+P`, `Transfer(R+P,D)` y parche de control.
- Exportación de parche, receta y reporte reproducible.
- Reversión al receptor original.

**Meta del spike:** probar que la arquitectura del cirujano funciona. **Meta de investigación posterior:** demostrar si su cirugía aporta mejora medible sobre controles fuertes.
