# MODEL MAXXING
## Spike técnico de viabilidad — Transferencia de capacidades entre LLMs

**Proyecto:** Model Maxxing  
**Versión:** 0.1 — Spike inicial  
**Duración:** 48 horas de trabajo  
**Tipo:** Prueba de concepto experimental  
**Contexto:** Trabajo de Fin de Grado — Ingeniería del Software  
**Estado:** Pendiente de ejecución

---

# 1. Objetivo

Model Maxxing es una plataforma experimental para potenciar capacidades específicas de modelos de lenguaje pequeños mediante información procedente de modelos donantes especializados.

La visión completa incluye tres roles:

- **Receptor (R):** modelo pequeño que queremos mejorar.
- **Donante (D):** modelo que domina mejor una capacidad objetivo.
- **LLM frontera:** agente que diagnostica los modelos, propone modificaciones sobre sus parámetros y dirige los experimentos.

El resultado deseado es un checkpoint modificado del receptor que funcione de forma autónoma, sin necesitar al donante ni al LLM frontera durante la inferencia.

## Objetivo de este spike

**Determinar si existe un escenario de transferencia de capacidades no trivial, medible y reproducible que justifique construir Model Maxxing.**

No vamos a desarrollar todavía la plataforma completa, la interfaz ni el agente autónomo.

Primero necesitamos demostrar que existe un problema real que el sistema pueda intentar resolver.

## Pregunta principal

> ¿Podemos mejorar una habilidad de un modelo receptor utilizando información de un donante especializado, conservando una capacidad propia del receptor que se perdería si simplemente lo sustituyéramos por el donante?

## Preguntas secundarias

1. ¿Son compatibles los checkpoints seleccionados?
2. ¿Existe una ventaja funcional real del donante sobre el receptor?
3. ¿El receptor tiene una especialidad que merece la pena proteger?
4. ¿La interpolación o transferencia de deltas produce candidatos funcionales?
5. ¿Existe un compromiso entre habilidad objetivo y habilidad protegida?
6. ¿Hay margen para que un algoritmo de transferencia selectiva supere los baselines sencillos?

---

# 2. Hipótesis

## H1 — Brecha funcional

El donante supera al receptor en una habilidad objetivo concreta.

Ejemplo:

`Score_Python(D) > Score_Python(R)`

## H2 — Especialidad protegida

El receptor supera al donante en otra capacidad que queremos conservar.

Ejemplo:

`Score_Medicina(R) > Score_Medicina(D)`

Esta condición evita que la solución óptima sea simplemente utilizar el donante completo.

## H3 — Transferencia parcial

Existe al menos una intervención sobre los parámetros del receptor que aumenta su rendimiento en la habilidad objetivo sin degradar excesivamente la capacidad protegida.

## H4 — Espacio de optimización no trivial

Diferentes intervenciones producen compromisos distintos entre las capacidades.

Si todos los candidatos son equivalentes o el donante completo domina claramente al receptor, no hay un problema interesante de optimización para el agente.

**Importante:** H3 y H4 son hipótesis que debemos comprobar, no hechos establecidos.

---

# 3. Modelos iniciales

## Configuración preferida

| Rol | Modelo |
|---|---|
| Base B | Qwen2.5-3B |
| Receptor R | Qwen2.5-3B especializado en medicina |
| Donante D | Qwen2.5-Coder-3B |
| Habilidad objetivo | Programación Python |
| Capacidad protegida | Conocimiento médico |

Enlaces de referencia:

- Base: https://huggingface.co/Qwen/Qwen2.5-3B
- Donante: https://huggingface.co/Qwen/Qwen2.5-Coder-3B

El receptor puede ser el modelo médico ya utilizado en los experimentos anteriores, siempre que tengamos acceso al checkpoint y podamos identificar su procedencia.

**No asumir que estos modelos son automáticamente compatibles para sumar deltas.**

La relación de entrenamiento entre base y donante debe verificarse antes de utilizar aritmética de task vectors.

## Comprobaciones obligatorias

Registrar para B, R y D:

- Nombre y repositorio.
- Revisión o commit del checkpoint.
- Licencia aplicable.
- Arquitectura.
- Número de capas.
- Dimensión oculta.
- Número de cabezas de atención.
- Configuración de MLP.
- Vocabulario y tokenizador.
- Formas de todos los tensores.
- Tipo de datos.
- Procedencia del ajuste especializado.
- Plantilla de chat, si corresponde.

Distinguir explícitamente modelos `Base` e `Instruct`.

No mezclar ambos tipos en un experimento principal sin controlar el efecto del ajuste de instrucciones.

## Condición para usar deltas

Para calcular:

`Delta_D = W_D - W_B`

deben verificarse, como mínimo, correspondencia de tensores, compatibilidad de tokenizador y una procedencia que haga razonable interpretar la diferencia como una modificación funcional transferible.

Que las matrices tengan la misma forma no demuestra por sí solo que el delta sea útil.

Si la procedencia no puede verificarse, la aritmética de deltas se marcará como exploratoria, y la interpolación directa se mantendrá como baseline independiente.

---

# 4. Alcance de las 48 horas

## Incluido

- Carga de modelos.
- Inspección de compatibilidad.
- Evaluación inicial.
- Análisis de diferencias de parámetros.
- Interpolación directa entre receptor y donante.
- Transferencia de delta cuando esté justificada.
- Comparación de resultados.
- Registro reproducible.
- Decisión Go / Pivot / Stop.

## Excluido

- Interfaz gráfica.
- Agente LLM autónomo completo.
- Transferencia 7B → 3B.
- Transferencia entre familias incompatibles.
- Entrenamiento o fine-tuning intensivo.
- Optimización de millones de máscaras.
- Benchmarking exhaustivo.
- Publicación de un modelo final.

El spike no debe convertirse en el desarrollo completo del TFG.

---

# 5. Experimento A — Baseline funcional

## Objetivo

Comprobar si el donante y el receptor tienen habilidades complementarias.

## Modelos

Evaluar:

1. Base B.
2. Receptor R.
3. Donante D.

## Benchmarks

### Programación

Preferencia:

- MBPP para diagnóstico y desarrollo.
- HumanEval reservado para una evaluación final posterior.

Para el spike, utilizar un subconjunto fijo y reproducible de problemas de MBPP, preferiblemente de 100 a 200 ejemplos si el tiempo de ejecución lo permite.

Registrar:

- Identificador del problema.
- Prompt exacto.
- Código generado.
- Resultado de las pruebas.
- Tiempo de generación.
- Errores de ejecución.

Las soluciones generadas deben ejecutarse en un entorno aislado, con límites de tiempo, memoria y acceso a red.

### Medicina

Utilizar un benchmark médico compatible con el receptor.

Posibles opciones:

- MedQA.
- Un conjunto médico previamente utilizado con el receptor.
- Preguntas médicas con respuestas verificables.

Para el spike, reservar un subconjunto fijo de aproximadamente 100 a 200 ejemplos, si está disponible.

No utilizar datos que hayan sido empleados para ajustar el receptor cuando pueda identificarse ese solapamiento.

## Tabla de resultados

| Modelo | Python | Medicina |
|---|---:|---:|
| B — Base | Pendiente | Pendiente |
| R — Receptor | Pendiente | Pendiente |
| D — Donante | Pendiente | Pendiente |

## Condiciones de interés

Queremos observar:

`Python(D) > Python(R)`

y simultáneamente:

`Medicina(R) > Medicina(D)`

Si no existe esta complementariedad, cambiar el par de modelos antes de continuar.

No basta con que las diferencias sean positivas: deben ser suficientemente grandes para justificar una transferencia y superar razonablemente la variabilidad de la evaluación.

---

# 6. Experimento B — Análisis de parámetros

## Objetivo

Comprobar la compatibilidad de los pesos y estudiar la estructura de las diferencias.

## Análisis

Para cada capa y módulo compatible, calcular:

`Delta = W_D - W_B`

y:

`relative_norm = ||Delta||_F / max(||W_B||_F, epsilon)`

Donde `epsilon` evita divisiones por cero.

Registrar estadísticas por:

- Capas de atención.
- Proyecciones Q, K, V y O.
- MLP.
- Embeddings.
- Normalizaciones.
- Cabeza de salida.

## Estadísticas recomendadas

- Norma de Frobenius.
- Norma relativa.
- Media y desviación típica de las diferencias.
- Porcentaje de parámetros con diferencia no nula.
- Valores máximos absolutos.
- Presencia de NaN o Inf.
- Similitud coseno entre tensores, cuando resulte interpretable.

## Visualizaciones

Generar:

1. Gráfico de norma relativa por capa.
2. Heatmap de diferencias por módulo.
3. Distribución de magnitudes del delta.
4. Resumen de incompatibilidades.

## Advertencia

Una norma de delta pequeña no demuestra que una habilidad esté localizada ni que el delta sea transferible.

Este análisis es diagnóstico, no una prueba de causalidad.

---

# 7. Experimento C — Interpolación directa

## Objetivo

Determinar si una mezcla sencilla de los modelos ya produce mejoras.

## Fórmula

Para checkpoints con correspondencia válida de tensores:

`W_candidate = (1 - alpha) * W_R + alpha * W_D`

## Coeficientes

Evaluar:

`alpha = [0.00, 0.25, 0.50, 0.75, 1.00]`

Interpretación:

- `alpha = 0`: receptor original.
- `alpha = 1`: donante completo.
- Valores intermedios: mezclas.

## Evaluación

Para cada candidato, ejecutar los mismos benchmarks y registrar:

| Alpha | Python | Medicina | Estado |
|---|---:|---:|---|
| 0.00 | Pendiente | Pendiente | Control |
| 0.25 | Pendiente | Pendiente | Pendiente |
| 0.50 | Pendiente | Pendiente | Pendiente |
| 0.75 | Pendiente | Pendiente | Pendiente |
| 1.00 | Pendiente | Pendiente | Control |

## Resultado buscado

Una mezcla intermedia que mejore programación y preserve mejor medicina que el donante completo.

Si el mejor candidato es siempre `alpha = 1` y no hay regresiones relevantes, el par no demuestra la necesidad de una transferencia selectiva.

---

# 8. Experimento D — Transferencia de delta

**Ejecutar únicamente si las comprobaciones de compatibilidad y procedencia lo justifican.**

## Fórmula

`Delta_D = W_D - W_B`

`W_candidate = W_R + alpha * Delta_D`

Evaluar inicialmente:

`alpha = [0.25, 0.50, 0.75, 1.00]`

Añadir un candidato con `alpha = 0` como control.

## Diferencia respecto a interpolación

La interpolación directa mezcla los pesos del receptor y del donante.

La transferencia de delta intenta añadir al receptor el cambio asociado al donante respecto a una base.

No son operaciones equivalentes cuando `R != B`.

## Evaluación

Comparar:

- Receptor original.
- Donante completo.
- Mejor interpolación.
- Mejor transferencia de delta.

## Criterio de interés

Que alguna transferencia mejore programación y conserve la especialidad médica dentro del límite definido.

No asumir que aplicar el delta completo será mejor que aplicar solo una parte.

---

# 9. Experimento E — Transferencia selectiva mínima

## Objetivo

Comprobar si existe una diferencia funcional entre transferir el delta completo y transferir solo determinados módulos.

Este experimento es opcional si los anteriores consumen demasiado tiempo.

## Estrategias

Probar un conjunto pequeño de máscaras predefinidas:

1. Solo MLP.
2. Solo atención.
3. Primer tercio de capas.
4. Tercio central.
5. Último tercio.

Usar un único valor de `alpha` seleccionado a partir de los experimentos anteriores.

## Fórmula

`W_candidate = W_R + alpha * (M * Delta_D)`

Donde `M` representa la máscara de parámetros seleccionados.

## Límite

Máximo cinco candidatos adicionales durante el spike.

No realizar una búsqueda exhaustiva por capas.

## Pregunta

¿Las regiones seleccionadas producen ganancias y regresiones diferentes?

Si la respuesta es afirmativa, existiría un espacio de decisiones que el futuro LLM frontera podría explorar.

---

# 10. Restricciones de rendimiento

Definir antes de ejecutar las evaluaciones:

## Capacidad objetivo

Maximizar rendimiento en programación.

## Capacidad protegida

Limitar la pérdida de rendimiento médico respecto al receptor original.

## Tolerancia inicial

`epsilon = 5 puntos porcentuales`

Condición:

`Medicina(candidate) >= Medicina(R) - 5 pp`

Esta tolerancia es provisional y deberá justificarse según la métrica y la variabilidad observada.

## Restricción estructural

El modelo final debe conservar:

- La arquitectura del receptor.
- Su número de parámetros.
- La capacidad de ejecutarse sin el donante.
- Un checkpoint cargable y válido.

## Función de selección

Entre los candidatos que cumplen las restricciones:

`best = argmax Python(candidate)`

Si ningún candidato modificado cumple las restricciones, el receptor original sigue siendo la solución válida.

---

# 11. Diseño de evaluación

## Separación de datos

Crear tres particiones:

- **Diagnóstico:** conocer diferencias entre modelos.
- **Selección:** comparar candidatos del spike.
- **Test final:** reservado para una evaluación posterior, sin utilizarlo para seleccionar parámetros.

El test final no debe utilizarse durante la búsqueda.

## Generación

Mantener constantes:

- Temperatura.
- Top-p.
- Número máximo de tokens.
- Plantilla de prompt.
- Criterio de extracción de respuesta.
- Configuración de ejecución.
- Semillas, cuando proceda.

Para una primera comparación, priorizar generación determinista si la configuración lo permite.

## Estadística

Registrar resultados por problema, no solo promedios.

Usar:

- Diferencias pareadas.
- Bootstrap pareado por problema.
- Intervalos de confianza.
- Análisis de ejemplos que cambian de incorrecto a correcto y viceversa.

Los subconjuntos pequeños del spike sirven para detectar señales y problemas de viabilidad, no para demostrar mejoras definitivas de pocos puntos.

No declarar una mejora científicamente concluyente basándose en una única diferencia pequeña.

---

# 12. Arquitectura mínima del repositorio

```text
model-maxxing/
├── README.md
├── pyproject.toml
├── configs/
│   ├── models.yaml
│   ├── benchmarks.yaml
│   └── spike.yaml
├── src/
│   └── model_maxxing/
│       ├── models/
│       │   ├── loader.py
│       │   └── compatibility.py
│       ├── profiling/
│       │   └── weight_analysis.py
│       ├── transfer/
│       │   ├── interpolation.py
│       │   ├── task_vector.py
│       │   └── masks.py
│       ├── evaluation/
│       │   ├── runner.py
│       │   └── metrics.py
│       └── experiments/
│           ├── registry.py
│           └── pipeline.py
├── scripts/
│   ├── inspect_models.py
│   ├── evaluate_baselines.py
│   ├── analyze_deltas.py
│   ├── sweep_interpolation.py
│   └── sweep_task_vector.py
├── tests/
│   ├── test_compatibility.py
│   ├── test_interpolation.py
│   └── test_task_vector.py
├── results/
│   ├── metrics/
│   ├── plots/
│   └── reports/
└── docs/
    └── SPIKE_REPORT.md
```

Los archivos indicados son la estructura que debe implementar el agente de programación. No se asume que ya existan.

---

# 13. Configuración del experimento

Ejemplo de `configs/spike.yaml`:

```yaml
project: model-maxxing
experiment: spike-001

models:
  base: Qwen/Qwen2.5-3B
  receiver: LOCAL_MEDICAL_QWEN_CHECKPOINT
  donor: Qwen/Qwen2.5-Coder-3B

objectives:
  target: python
  protected:
    - medical_qa

constraints:
  max_protected_loss_pp: 5.0
  preserve_receiver_architecture: true
  standalone_inference: true

evaluation:
  coding_dataset: mbpp
  medical_dataset: medical_qa
  diagnostic_sample_size: 100
  selection_sample_size: 100
  seed: 42

experiments:
  interpolation:
    enabled: true
    alphas: [0.0, 0.25, 0.5, 0.75, 1.0]

  task_vector:
    enabled: auto_if_compatible
    alphas: [0.0, 0.25, 0.5, 0.75, 1.0]

  selective_transfer:
    enabled: optional
    max_candidates: 5

output:
  directory: results/
  save_full_checkpoints: false
  save_recipes: true
```

Los tamaños de muestra son valores iniciales configurables. Deben adaptarse al tiempo disponible y al número de ejemplos realmente disponibles en cada partición.

---

# 14. Requisitos de implementación

## Precisión numérica

- Aplicar operaciones de mezcla en FP32 cuando sea necesario para estabilidad.
- Convertir después al dtype de almacenamiento elegido.
- Verificar NaN e Inf.
- No modificar los checkpoints originales.
- Validar el checkpoint resultante cargándolo antes de evaluarlo.

## Gestión de memoria

No mantener todos los modelos y candidatos simultáneamente en VRAM.

Preferir:

- Operaciones por tensores o fragmentos.
- Lectura secuencial de checkpoints.
- Evaluación de un candidato cada vez.
- Liberación de recursos entre experimentos.
- Guardado de recetas en lugar de múltiples copias completas cuando sea posible.

Como referencia, un modelo de 3B en FP16 ocupa alrededor de 6 GB solo en pesos. La memoria real necesaria para ejecutar y transformar modelos será superior.

## Persistencia

Guardar por experimento:

- Identificador.
- Fecha.
- Modelos y revisiones.
- Configuración.
- Operador aplicado.
- Parámetros de la operación.
- Semilla.
- Resultados por benchmark.
- Resultados por problema.
- Tiempo de ejecución.
- Estado y errores.

## Seguridad

- No ejecutar código generado por modelos directamente en el sistema anfitrión.
- Utilizar un entorno aislado para benchmarks de programación.
- Limitar CPU, memoria, tiempo y red.
- Evitar carga de código remoto no auditado.
- Comprobar integridad y procedencia de checkpoints.

---

# 15. Tests mínimos

## Test 1 — Identidad de interpolación

Para `alpha = 0`:

`interpolate(R, D, 0) == R`

Para `alpha = 1`:

`interpolate(R, D, 1) == D`

Comparar tensores con tolerancias numéricas apropiadas.

## Test 2 — Identidad de delta

Para `alpha = 0`:

`apply_delta(R, Delta, 0) == R`

## Test 3 — Máscara vacía

Una máscara completamente nula no debe modificar el receptor.

## Test 4 — Incompatibilidad

El sistema debe rechazar una operación cuando las formas de tensores o configuraciones no sean compatibles.

## Test 5 — Integridad

El candidato no debe contener valores NaN o Inf y debe poder cargarse.

## Test 6 — Reproducibilidad

La misma receta aplicada a los mismos checkpoints debe producir resultados numéricamente equivalentes bajo las condiciones documentadas.

---

# 16. Cronograma de 48 horas

## Día 1 — Compatibilidad y baseline

### Horas 0–2

- [ ] Crear repositorio.
- [ ] Configurar entorno.
- [ ] Registrar versiones de dependencias.
- [ ] Identificar checkpoints.
- [ ] Comprobar licencias.
- [ ] Guardar manifiesto de modelos.

### Horas 2–5

- [ ] Implementar inspección de arquitectura.
- [ ] Comparar tensores.
- [ ] Verificar tokenizadores.
- [ ] Identificar procedencia.
- [ ] Decidir si se permite task vector.

### Horas 5–9

- [ ] Preparar benchmarks.
- [ ] Ejecutar receptor.
- [ ] Ejecutar donante.
- [ ] Ejecutar base si procede.
- [ ] Guardar resultados por problema.

### Horas 9–12

- [ ] Analizar complementariedad.
- [ ] Calcular estadísticas de deltas.
- [ ] Generar primeros gráficos.
- [ ] Decidir si continuar con el par seleccionado.

**Entrega del día 1:** informe de compatibilidad y tabla de rendimiento inicial.

## Día 2 — Transferencia y decisión

### Horas 12–16

- [ ] Implementar interpolación.
- [ ] Ejecutar barrido de coeficientes.
- [ ] Validar candidatos.
- [ ] Evaluar programación y medicina.

### Horas 16–20

- [ ] Implementar task vector si procede.
- [ ] Ejecutar barrido.
- [ ] Registrar métricas.
- [ ] Comparar con interpolación.

### Horas 20–23

- [ ] Probar máscaras simples si queda tiempo.
- [ ] Generar gráficos.
- [ ] Calcular diferencias pareadas.
- [ ] Identificar mejores candidatos.

### Horas 23–24

- [ ] Redactar informe final.
- [ ] Aplicar criterios Go / Pivot / Stop.
- [ ] Proponer siguiente experimento.
- [ ] Guardar configuraciones reproducibles.

El cronograma representa aproximadamente 24 horas de trabajo efectivo distribuidas en dos días. Las evaluaciones de GPU pueden continuar mientras se realizan otras tareas.

Si la ejecución supera el presupuesto, priorizar baselines e interpolación antes que las máscaras selectivas.

---

# 17. Criterios de decisión

## GO — Continuar con Model Maxxing

Continuar si:

- El donante supera claramente al receptor en la habilidad objetivo.
- El receptor conserva una especialidad propia que el donante no domina.
- Los modelos permiten al menos una operación de transferencia válida.
- Se observa alguna mejora útil bajo las restricciones o un compromiso funcional claro entre candidatos.
- Los resultados son reproducibles a nivel de pipeline.

**Consecuencia:** comenzar el desarrollo del motor de búsqueda adaptativa y del agente LLM frontera.

## PIVOT — Cambiar de modelos o estrategia

Cambiar el diseño si:

- El donante no presenta una ventaja funcional.
- El receptor no tiene una capacidad diferenciada que proteger.
- El donante completo domina todas las métricas.
- La procedencia de los checkpoints invalida la aritmética de deltas.
- Las interpolaciones fallan, pero existe otra familia de operaciones plausible.
- El benchmark no distingue suficientemente a los modelos.

Posibles pivotes:

1. Cambiar de receptor.
2. Cambiar de donante.
3. Cambiar de habilidad objetivo.
4. Utilizar modelos derivados de una base mejor documentada.
5. Estudiar transferencia mediante activaciones o destilación como alternativa.

## STOP — No avanzar con esta hipótesis

Detener la ruta experimental concreta si:

- No aparece complementariedad entre modelos tras probar pares razonables.
- No existen operaciones de transferencia viables con los recursos disponibles.
- Las modificaciones degradan sistemáticamente las capacidades.
- Los resultados no pueden reproducirse.
- El coste impide realizar evaluaciones mínimamente fiables.

**STOP no significa cancelar automáticamente todo el TFG.**

La plataforma puede pivotar hacia una herramienta de evaluación, compatibilidad y comparación reproducible de métodos de transferencia, pero habría que reformular su pregunta investigadora.

---

# 18. Métricas de éxito del spike

## Métrica principal

Ganancia en la habilidad objetivo:

`Gain = Score_Python(candidate) - Score_Python(R)`

## Regresión protegida

`Loss = Score_Medicine(R) - Score_Medicine(candidate)`

## Viabilidad

Un candidato es válido cuando:

`Gain > 0`

y:

`Loss <= epsilon`

La mejora debe analizarse con su incertidumbre estadística.

## Métricas adicionales

- Número de candidatos válidos.
- Mejor ganancia observada.
- Mayor regresión observada.
- Tiempo por experimento.
- Consumo máximo de VRAM.
- Espacio en disco.
- Número de fallos de ejecución.
- Reproducibilidad de resultados.
- Diferencia frente a reemplazar el receptor por el donante.

---

# 19. Entregables

Al finalizar el spike debe existir:

## Código

- Scripts de carga y compatibilidad.
- Implementación de interpolación.
- Implementación de deltas, si procede.
- Pipeline de evaluación.
- Tests mínimos.

## Resultados

- `compatibility.json`
- `baseline_metrics.csv`
- `interpolation_results.csv`
- `task_vector_results.csv` si procede
- `per_example_results.csv`
- Gráficos de rendimiento.
- Gráficos de normas de deltas.
- Configuraciones y semillas.

## Informe

Archivo `docs/SPIKE_REPORT.md` con:

1. Objetivo.
2. Hardware utilizado.
3. Modelos y revisiones.
4. Compatibilidad.
5. Benchmarks.
6. Resultados.
7. Análisis de errores.
8. Limitaciones.
9. Decisión Go / Pivot / Stop.
10. Próximos pasos.

No inventar resultados. Si un experimento no se ha podido ejecutar, debe aparecer explícitamente como no realizado.

---

# 20. Qué viene después si el spike funciona

El siguiente paso será construir el núcleo inteligente de Model Maxxing.

## Fase 1 — Model Profiler

Generar diagnósticos estructurados de diferencias funcionales y de parámetros.

## Fase 2 — Transfer Engine

Ampliar los operadores disponibles:

- Máscaras por módulos.
- Selección de capas.
- Intensidades independientes.
- Restauración selectiva.
- Composición de parches.

## Fase 3 — Frontier Engineer

Conectar un LLM frontera que pueda consultar herramientas y proponer experimentos.

Ejemplos de operaciones:

```python
inspect_model(model_id)
compare_models(receiver, donor)
inspect_delta(base, donor, modules)
propose_transfer(operator, parameters)
evaluate_candidate(candidate_id)
compare_candidates(candidate_ids)
rollback(candidate_id)
```

## Fase 4 — Comparación experimental

Comparar el agente con:

- Búsqueda aleatoria.
- Barridos sistemáticos.
- Optimización convencional.
- Técnicas de merging existentes.

Los métodos deben disponer del mismo espacio de operaciones, presupuesto y datos comparables.

## Fase 5 — Producto

Construir la plataforma:

- Importación de modelos.
- Configuración del overclock.
- Gestión de experimentos.
- Visualización de benchmarks.
- Historial de modificaciones.
- Exportación de checkpoints.
- Rollback.

La transferencia entre tamaños distintos, como 7B → 3B, se mantendrá como extensión hasta demostrar su viabilidad.

---

# 21. Instrucciones para el agente de programación

**Objetivo:** implementar y ejecutar el spike técnico de Model Maxxing descrito en este documento.

### Reglas obligatorias

1. No implementar una interfaz gráfica.
2. No entrenar modelos.
3. No asumir que los checkpoints comparten una base válida.
4. No modificar checkpoints originales.
5. No inventar métricas ni resultados.
6. No ejecutar código generado sin aislamiento.
7. No utilizar el conjunto de test final para seleccionar parámetros.
8. No lanzar experimentos costosos sin estimar recursos.
9. Guardar resultados por problema y configuración.
10. Priorizar la viabilidad experimental sobre la complejidad arquitectónica.

### Orden de trabajo

Primero verificar compatibilidad.

Después evaluar los modelos originales.

Solo si existe complementariedad, ejecutar interpolaciones y deltas.

Finalmente generar el informe y aplicar los criterios Go / Pivot / Stop.

Si falta un checkpoint, una licencia, un dataset o una dependencia crítica, documentar el bloqueo y solicitar la información necesaria antes de sustituir silenciosamente el experimento.

### Definición de terminado

El spike está terminado cuando podemos responder con datos a estas preguntas:

- ¿Qué sabe hacer mejor el donante?
- ¿Qué capacidad propia tiene el receptor?
- ¿Son compatibles para una transferencia?
- ¿Qué ocurre cuando mezclamos sus pesos?
- ¿Podemos mejorar el receptor sin destruir su especialidad?
- ¿Existe un problema de optimización suficientemente interesante para justificar un LLM frontera?
- ¿Qué deberíamos construir a continuación?

---

# 22. Conclusión

Model Maxxing aspira a convertirse en un entorno de ingeniería de capacidades para modelos de lenguaje.

Sin embargo, antes de construir un sistema inteligente de overclock necesitamos comprobar que existe una transferencia real que merezca ser optimizada.

Este spike no pretende demostrar que podemos instalar cualquier habilidad en cualquier modelo.

Pretende identificar un escenario experimental donde:

**Un donante aporta una capacidad, el receptor aporta otra y existe un compromiso real que podemos explorar mediante modificaciones controladas de parámetros.**

Si encontramos ese escenario, tendremos una base técnica sólida para desarrollar Model Maxxing como TFG de Ingeniería del Software.

**Primero demostrar que existe algo que optimizar. Después construir el sistema que lo optimice.**
