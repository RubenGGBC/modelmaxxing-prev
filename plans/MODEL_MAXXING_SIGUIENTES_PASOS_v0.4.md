# MODEL MAXXING — Plan de ejecución v0.4

**Documento para Claude Code · Continuación del MVP, sin reiniciar el repositorio**  
**Fecha:** 9 de noviembre de 2026  
**Estado de partida:** Spike v0.3 finalizado con decisión **PIVOT**  
**Hardware disponible:** PC con GPU NVIDIA de 8 GB de VRAM  
**Objetivo de esta fase:** comprobar si un LLM frontera puede **diseñar modificaciones reales de pesos, informadas por experimentos, que potencien una habilidad objetivo mejor que los métodos convencionales**, antes y después de una transferencia compatible.

> **Instrucción prioritaria:** no conviertas Model Maxxing en un optimizador de pipelines ni en un sistema cuya misión obligatoria sea preservar medicina. El usuario define qué habilidad quiere **maximizar** y qué pérdidas acepta. Claude debe actuar como **Weight Surgeon**, no solo como selector de algoritmos.

---

## 0. Definición congelada del proyecto

**Model Maxxing** es una plataforma de overclock de LLMs. El usuario aporta un **receptor** pequeño, un **donante** con una capacidad superior y un objetivo en lenguaje natural, por ejemplo:

> «Maximiza la generación de código Python de este modelo de 1.5B. Puedes sacrificar capacidades generales; quiero mantener el tamaño del modelo y ejecutarlo localmente».

El circuito central es:

```text
Prompt del usuario
       ↓
Claude traduce el objetivo a capacidades evaluables
       ↓
Claude diseña estímulos contrastivos para receptor y donante
       ↓
Diagnóstico funcional + activaciones + intervenciones experimentales
       ↓
Claude construye un parche NUMÉRICO restringido de pesos
       ↓
Receptor y/o donante preacondicionados
       ↓
Algoritmo de transferencia compatible
       ↓
Benchmark independiente + feedback + elección de checkpoint
       ↓
Modelo final autónomo (sin donante ni Claude en inferencia)
```

**Reglas del alcance actual:**

- **Sí:** cirugía previa de pesos, estímulos, diagnósticos, un operador de transferencia, comparaciones justas, recetas y rollback.
- **No:** flotas de agentes, Project Mode, interfaz compleja, múltiples donantes simultáneos, universalidad entre familias, entrenar un LLM frontera, crear CAST como algoritmo separado.
- **Preservar capacidades es opcional.** En modo `maxxing`, la métrica primaria es la habilidad objetivo. Mediremos degradaciones secundarias, pero solo serán restricciones si el usuario las configura.
- La cirugía puede aplicarse en receptor y, más adelante, donante. **Primero estudiar la intervención en el receptor**; no introducir ambos grados de libertad a la vez.

### Hipótesis principal

> Dado el mismo receptor, donante, operador de transferencia, evidencias disponibles, presupuesto de modificación y presupuesto de evaluaciones, **¿un parche construido por Claude mejora la habilidad objetivo más que un parche de una heurística o de un optimizador numérico?**

La hipótesis aún no está demostrada. El éxito técnico (aplicar y revertir pesos) y el éxito científico (mejorar frente a controles) se evalúan por separado.

---

## 1. Cerrar y archivar v0.3 — NO volver a repetirlo completo

Ya tenemos infraestructura de parches, validación, estadísticas, diagnósticos y evaluaciones. Conservarla.

### Resultados v0.3 de referencia

| Variante | MBPP / 100 | NLL código | Lectura |
|---|---:|---:|---|
| R original | 38 | 1.2996 | Baseline |
| Transferencia sin cirugía `A_T` | 45 | 1.3006 | Control obligatorio |
| Claude sin transferencia `D` | 41 | 1.2840 | Efecto medible en NLL |
| Claude + transferencia `D_T` | 43 | 1.2882 | No mejora frente a `A_T` |
| Heurística fija `C` | 41 | 1.2869 | Prácticamente empata con Claude |
| `0.5·R + 0.5·B` | **49** | **1.2440** | Control de retorno a base, dominante |

**Interpretación correcta:** el parche de Claude es real y altera probabilidades; **no hay ventaja atribuible a Claude** frente a la regla fija. El par de 0.5B tiene poca brecha en programación y el receptor puede estar perjudicado por su ajuste. No convertir la mejora al volver a la base en una prueba de transferencia desde el donante.

### Acciones

- [ ] Crear un `git tag` para el estado v0.3 y conservar manifiestos, hashes de parches, commits y resultados por ejemplo.
- [ ] Ejecutar los tests actuales en el PC y registrar versión de Python, PyTorch/CUDA y GPU.
- [ ] Guardar el informe v0.3 como referencia inmutable. No reoptimizar sobre los 100 ejemplos de MBPP ya vistos.
- [ ] Marcar `v0.3 = PIVOT`, sin reescribir los resultados para que parezcan positivos.

**Entregable:** `docs/BASELINE_V03_LOCKED.md` con rutas existentes y limitaciones.

---

## 2. Puerta 1: elegir un experimento mejor con 1.5B

**Decisión:** 0.5B queda como banco de tests baratos. El experimento principal pasa a **1.5B**, si el hardware y la calidad del par lo permiten. 3B queda para validación posterior.

### Modelos candidatos (NO asumir que están validados)

- Base candidata: [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B).
- Donante candidato: [`Qwen/Qwen2.5-Coder-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B).
- Receptor: checkpoint de 1.5B compatible y **distinto del donante**, seleccionado según la capacidad que realmente deseamos potenciar. Puede ser generalista o especializado en otra área; **no hay requisito de medicina**.

**Advertencia:** si `R = B` y `D` resulta efectivamente un derivado alineado de `B`, una transferencia de delta completo `R + (D-B)` equivale al donante. Esto es útil para depurar, pero **no demuestra por sí mismo una cirugía superior**. Incluir siempre `D` completo como baseline si tiene el mismo tamaño y puede ejecutarse localmente.

### Filtro de selección (antes de empezar cirugías)

- [ ] Fijar repositorio y commit de cada modelo, licencia, arquitectura, tokenizer, plantilla de prompts y procedencia.
- [ ] Verificar formas de tensores y compatibilidad real de las operaciones. La coincidencia de tamaños no prueba alineamiento funcional.
- [ ] Evaluar R, D y B en un conjunto **de selección** no utilizado en v0.3.
- [ ] Medir si D supera a R en **subtipos identificables** de programación (p. ej. recursión, depuración, transformaciones de listas), no solo en un promedio.
- [ ] Evaluar D directamente: si D gana claramente y no existe ventaja de usar R, dejar constancia de que **usar D es el producto óptimo para ese caso**.
- [ ] Registrar NLL, pass@1, latencia y memoria; evitar basar decisiones en diferencias de 1–2 problemas.

**GO a cirugía:** hay una brecha funcional clara y un procedimiento de transferencia compatible que deja margen para mejorar un resultado de referencia.  
**PIVOT de modelos:** el donante no mejora la habilidad objetivo, los checkpoints no son razonablemente compatibles, o reemplazar por D resuelve todo sin beneficio de cirugía.  
**STOP del experimento, no del TFG:** no se puede construir un benchmark fiable o reproducible con los recursos actuales.

> No ampliar a 7B→3B para arreglar un mal par 1.5B: eso introduciría un segundo problema (alineamiento entre tamaños) antes de evaluar la hipótesis del cirujano.

---

## 3. Puerta 2: Causal Weight Surgeon v1 (objetivo principal)

No basta con entregar a Claude una tabla de «capas buenas/malas». El agente debe **producir sus propias hipótesis apoyándose en pruebas funcionales** y limitar sus intervenciones a regiones justificadas. No asumir que las habilidades están localizadas en una neurona concreta.

### 3.1 Traducir el prompt a una especificación de capacidad

Entrada:

```yaml
user_goal: "Maximiza Python, acepto sacrificar otras habilidades"
target_capability: python_code_generation
primary_metric: pass_at_1
secondary_metrics: [code_nll, medical_accuracy, general_nll]
secondary_metrics_are_constraints: false
standalone_inference: true
keep_receiver_architecture: true
experiment_budget: 24  # ejemplo, fijar según recursos
```

El número de pruebas es configurable. El agente **no puede cambiar la métrica primaria después de observar resultados**.

### 3.2 Diseñar estímulos contrastivos

Claude recibe ejemplos R-falla/D-acierta y genera estímulos que intentan aislar la habilidad buscada. Por ejemplo:

- Una implementación recursiva y su equivalente iterativo.
- El mismo algoritmo con casos límite más difíciles.
- Código correcto frente a una variante con un bug controlado.
- Prompts equivalentes que cambian estructura sin cambiar solución.

Cada estímulo debe registrar `hypothesis_id`, `stimulus_id`, `family`, `expected_contrast`, prompt, respuesta y resultado del validador.

**Separación estricta:** estímulos y pruebas de diagnóstico no pueden provenir del conjunto de test final; tampoco se mostrarán al agente problemas o respuestas del test final.

### 3.3 Medir evidencias, no solo activaciones

Herramientas a implementar o ampliar:

```python
compare_behavior(receiver, donor, stimulus_ids)
collect_activations(model, stimulus_ids, layers, modules)
rank_regions_by_intervention(model, stimulus_ids, allowed_modules)
inspect_delta(base, donor, regions)
inspect_gradients_or_sensitivity(model, stimulus_ids, regions)
inspect_history(experiment_id)
```

**Prioridad causal:** usar intervenciones **dentro del receptor** (ablation/activation replacement controlado) y medir cambios en una métrica funcional predefinida. Las diferencias de activación entre modelos son evidencia exploratoria, no causal por sí solas. El *patching* de activaciones entre receptor y donante solo se permite si se justifica su alineamiento y compatibilidad.

Para cada región candidata guardar: ejemplos afectados, efecto positivo/negativo, controles de intervención, varianza entre estímulos y confianza limitada de la hipótesis.

### 3.4 Construir un espacio quirúrgico restringido

Solo abrir regiones respaldadas por evidencia. Primera configuración sugerida (sujeta a pruebas de presupuesto):

- Máximo **4 tensores** por candidato; comenzar por `mlp.down_proj` u otros módulos con señal consistente.
- Parche de rango **≤ 4**; ampliar a 8 solo si la sensibilidad justifica más libertad.
- Presupuesto de norma **relativo al tensor**, predefinido y comparable entre brazos. Evaluar unos pocos niveles en desarrollo, pero no elegirlo utilizando el test final.
- Usar una base reproducible `U, V` obtenida de SVD, activaciones o gradientes del **conjunto de diagnóstico**.
- Claude propone la pequeña matriz de coeficientes `C`:

\[
P = U C V^{\top},\qquad R^*=R+P.
\]

**Importante:** Claude debe generar una transformación numérica que vaya **más allá de elegir un escalar o copiar la primera dirección SVD**. La salida es una receta tipada; no ejecutar código arbitrario generado por el agente.

### 3.5 Contrato obligatorio de propuesta

```json
{
  "hypothesis_id": "hyp_0001",
  "evidence_ids": ["stim_12", "patch_test_08"],
  "target_model": "receiver",
  "target_tensor": "model.layers.12.mlp.down_proj.weight",
  "operator": "low_rank_coefficients",
  "basis_u_id": "basis_u_12",
  "basis_v_id": "basis_v_12",
  "coefficients": [[0.12, -0.03], [0.01, 0.08]],
  "expected_effect": "reduce code NLL on recursion diagnostics",
  "falsification_test": "held-out contrastive recursion set",
  "max_relative_norm": 0.005
}
```

Esquema **ilustrativo**, no una receta lista para ejecutar: el validador debe comprobar dimensiones, rango, correspondencia de identificadores, presupuesto de norma, lista blanca, finitud y no alteración de originales.

El agente debe guardar antes de evaluar: evidencia consultada, hipótesis, parche exacto y predicción concreta. No aceptar justificaciones post hoc.

---

## 4. Puerta 3: una sola transferencia primero

El estudio principal utiliza **un único operador de transferencia** compatible con los modelos escogidos, idéntico en todos los brazos. Comenzar por task vector o una fusión simple cuando la procedencia lo permita:

\[
R_{\mathrm{final}} = \operatorname{Transfer}(R + P, D).
\]

Si se aplica `R + λ(D−B)`, comprobar que la modificación `P` no se confunde con el efecto de `−Δ_R` y que el donante aporta señal propia. Si se usa interpolación `(1−α)(R+P)+αD`, el parche queda escalado por `(1−α)` y desaparece en `α=1`.

- [ ] Primero verificar efectos del parche **sin transferencia**.
- [ ] Después aplicar exactamente la misma transferencia a todos los candidatos.
- [ ] Mantener `R→B` como control en casos con receptor ajustado, para detectar mera recuperación del modelo base.
- [ ] Introducir TIES/DARE solo como **segunda comparación** si el primer operador funciona y la compatibilidad está verificada. TIES está pensado principalmente para resolver conflictos al combinar deltas; no añadirlo solo por llenar la lista de algoritmos.

**No** evaluar 10 algoritmos, cambiar el donante y dejar que Claude modifique también el donante en el mismo experimento: sería imposible atribuir las diferencias.

---

## 5. Diseño experimental pre-registrado

### Brazos mínimos

| Brazo | Descripción | ¿Parche construido por Claude? |
|---|---|---|
| `R` | Receptor original | No |
| `D` | Donante completo, cuando tamaño/uso comparable | No |
| `T` | Transferencia convencional `Transfer(R,D)` | No |
| `BASE` | Receptor interpolado hacia B, cuando proceda | No |
| `NUM` | Parche numérico optimizado sin LLM con la **misma evidencia** y presupuesto | No |
| `HEUR` | Regla fija basada en diagnósticos | No |
| `RAND` | Control aleatorio, misma región, rango y norma; varias semillas | No |
| `CLAUDE` | Parche que Claude construye en el espacio permitido | Sí |
| `CLAUDE+T` | Cirugía de Claude seguida del mismo operador T | Sí |

Cuando proceda, evaluar también `NUM+T`, `HEUR+T`, `RAND+T` y parche Claude sin transferencia. Para limitar coste, usar una fase de descarte por NLL y reservar las ejecuciones de pass@1 completas para candidatos preseleccionados según reglas **fijadas antes**.

### Equidad experimental

- **Mismo conjunto de estímulos disponible** para Claude y controles, o justificar y medir explícitamente la ventaja informacional.
- **Mismos módulos y bases `U,V`**, norma, rango máximo y número de tensores.
- **Mismo número de evaluaciones/propuestas** y registrar además horas de GPU y coste de API. Si el método numérico necesita más cómputo para generar candidatos, informar ambas magnitudes.
- **Mismo operador de transferencia** y mismos parámetros posteriores para cada comparación.
- Guardar la receta y hash de Claude **antes** de cualquier evaluación de selección final.
- Repetir los controles aleatorios con varias semillas; registrar cada resultado, no solo el mejor.

### Datos y métricas

| Partición | Uso | Visible para Claude |
|---|---|---|
| Diagnóstico | Generar estímulos, localizar regiones y diseñar hipótesis | Sí |
| Desarrollo/selección | Elegir intervención dentro del presupuesto | Solo resultados agregados permitidos |
| Test final independiente | Comparar métodos después de congelar diseños | **No** |

Los primeros 100 ejemplos MBPP ya usados repetidamente en v0.3 **no son test final**. Usar otros problemas no vistos y revisar contaminación de datos en modelos y donantes. Opciones: remanente no consultado de MBPP, HumanEval/EvalPlus si no hubo exposición, y una colección nueva de tareas de código evaluables mediante tests. Con muestras pequeñas, informar intervalos de confianza amplios y evitar afirmaciones concluyentes.

**Métrica primaria:** pass@1 en habilidad elegida y conjunto fijado.  
**Métricas auxiliares:** NLL sobre código, tasa de sintaxis válida, fallos por clase, latencia, VRAM, tiempo y coste total.  
**Otras capacidades:** medir, pero **no restringir** en modo `maxxing` salvo solicitud del usuario.

Usar bootstrap pareado por problema y publicar la tabla de aciertos por ID. Predefinir un efecto mínimo de interés antes de conocer resultados nuevos; no tratar una mejora aparente de 1–2 problemas como prueba científica.

---

## 6. Puertas de decisión

### GO técnico

- [ ] El usuario puede especificar la habilidad objetivo mediante un prompt.
- [ ] Claude genera estímulos y recibe evidencias estructuradas.
- [ ] Claude produce un parche numérico real dentro de una whitelist.
- [ ] Se aplica y revierte sin alterar checkpoints originales.
- [ ] Funciona con el operador de transferencia y se exporta el modelo autónomo.
- [ ] El experimento puede reconstruirse desde una receta y revisiones fijadas.

### GO experimental (más exigente)

- [ ] Donante con ventaja útil en habilidad objetivo y par compatible.
- [ ] Parche `CLAUDE+T` superior a `T` y a controles de presupuesto equivalente en el **test reservado**; informar tamaño del efecto e intervalo de confianza.
- [ ] La mejora no se explica simplemente por acercarse a la base o usar el donante completo.
- [ ] Resultado razonablemente estable en nuevas tareas y, cuando sea viable, un segundo par de modelos o más de una ejecución del agente.

### PIVOT

Si la cirugía no supera controles:

1. Comprobar si los estímulos identifican realmente una diferencia funcional aprovechable.
2. Aislar si el problema está en la **localización**, en el **parche** o en la **transferencia posterior**.
3. Comparar un optimizador numérico sobre las mismas bases con Claude, sin cambiar simultáneamente modelo, operador y dataset.
4. Si la mejora solo aparece al volver a B, estudiarlo como recuperación de capacidades, **no** como transferencia desde D.
5. Si el donante no es claramente mejor, cambiar **solo el par**, manteniendo protocolo y métricas.

### STOP o reformulación

Si no se observan mejoras robustas tras cambios controlados, concluir honestamente que no hay evidencia a favor de la cirugía de Claude bajo las condiciones estudiadas. Conservar la **plataforma de ingeniería** como resultado y documentar límites y costes. No añadir funcionalidades ajenas al experimento para ocultar un resultado nulo.

---

## 7. Orden exacto de implementación para Claude Code

### Hito A — Preparar ejecución 1.5B

- [ ] Confirmar CUDA, VRAM real, RAM y disco; revisar Docker aislado para los tests de código.
- [ ] Asegurar carga secuencial: no mantener B, R y D completos en GPU a la vez.
- [ ] Preservar los `18 tests` existentes; añadir checks de memoria y de consistencia de dtypes.
- [ ] Evaluar candidatos de modelo y documentar la selección en `docs/MODEL_SELECTION.md`.

**Salida:** baseline confiable de R/B/D y par elegido con justificación.

### Hito B — Estimulación

- [ ] Implementar `goal_to_capability_spec`.
- [ ] Implementar `stimulus_generator` y validador de familias contrastivas.
- [ ] Comparar aciertos/fallos de R y D y guardar ejemplos por ID.

**Salida:** `results/diagnostics/stimuli_manifest.json` y `capability_profile.json`.

### Hito C — Localización y cirugía

- [ ] Extender diagnósticos a intervenciones temporales y controles negativos.
- [ ] Derivar bases `U,V` reproducibles y acotar módulos.
- [ ] Extender `frontier_surgeon.py` para que Claude devuelva `C` y su hipótesis, no solo una máscara o `alpha`.
- [ ] Validar, guardar hash y bloquear parche antes del test independiente.

**Salida:** `configs/surgery_claude_locked.json`, parches y pruebas nuevas de validación.

### Hito D — Comparación

- [ ] Ejecutar R, D, T, BASE (si aplica), NUM, HEUR, RAND y CLAUDE; evaluar variantes `+T` equivalentes.
- [ ] Repetir RAND con varias semillas; registrar coste y resultados por ejemplo.
- [ ] Mantener métricas secundarias visibles, sin penalizarlas si el perfil es `maxxing`.

**Salida:** `results/experiments/maxxing_v04/` y `docs/EXPERIMENT_V04.md`.

### Hito E — Demo mínima de ingeniería

- [ ] Un CLI o endpoint que acepte un objetivo, modelo receptor, donante y presupuesto.
- [ ] Una ejecución completa reanudable con logs, recetas y rollback.
- [ ] Exportación de checkpoint con inferencia independiente.
- [ ] Informe final separando contribución de software y hallazgo de investigación.

**Salida:** demo reproducible ejecutable sin interfaz React. La UI queda para después de validar el circuito.

---

## 8. Estructura de repositorio: evolución mínima

Aprovechar archivos existentes. Añadir solo lo necesario:

```text
src/model_maxxing/
  agents/
    frontier_surgeon.py        # ampliar: hipótesis + coeficientes del parche
    stimulus_designer.py       # NUEVO: pruebas contrastivas desde objetivo
  diagnostics/
    causal_regions.py          # NUEVO: intervenciones temporales/controladas
    capability_gap.py          # NUEVO: R falla / D acierta por subhabilidad
  surgery/
    arms.py                    # reutilizar + comparadores equivalentes
    builder.py                 # reutilizar
    validator.py               # reutilizar
  transfer/
    engine.py                  # reutilizar, 1 operador principal
  evaluation/
    runner.py                  # reutilizar
    metrics.py                 # reutilizar, bootstrap pareado
  experiments/
    pipeline.py                # orquestación, presupuesto, reanudación
scripts/
  select_models.py
  build_stimuli.py
  run_weight_surgeon.py
  compare_arms.py
configs/
  maxxing_v04.yaml
  surgery_claude_locked.json
results/experiments/maxxing_v04/
docs/
  BASELINE_V03_LOCKED.md
  MODEL_SELECTION.md
  EXPERIMENT_V04.md
```

No renombrar ni reescribir código existente que ya pasa tests salvo necesidad justificada. La estructura es orientativa: adaptar a los nombres reales del repositorio.

---

## 9. Límites de seguridad y reproducibilidad

- Ejecutar código generado en sandbox aislado con límites de tiempo, memoria, CPU y red; si Docker no está operativo, fallar de manera controlada.
- No ejecutar libremente scripts Python emitidos por Claude sobre checkpoints. Usar recetas declarativas y validadores estrictos.
- No editar originales; fijar hashes, commits y parámetros de entorno.
- Calcular y guardar deltas/parches con dtypes controlados; probar identidad, norma, finitud, persistencia y rollback.
- Medir memoria real. Los pesos de un 1.5B en FP16/BF16 ocupan ~3 GB, sin contar cachés, activaciones, otros checkpoints ni buffers.
- Si las modificaciones exceden 8 GB de VRAM, hacerlas por fragmentos en CPU/RAM y evaluar un candidato cada vez.
- Revisar licencias antes de distribuir checkpoints derivados; un repositorio sin licencia declarada no autoriza automáticamente redistribución.

---

## 10. Instrucción final para Claude Code (copiar como objetivo de trabajo)

> Continúa **Model Maxxing** desde v0.3; no reinicies el repositorio. Archiva los resultados existentes y conserva los módulos probados. El siguiente objetivo es **Causal Weight Surgeon v1**: un usuario define una habilidad mediante prompt; Claude diseña estímulos contrastivos para el receptor y el donante, solicita diagnósticos y efectos de intervenciones, formula una hipótesis falsable y **construye una modificación numérica propia de pesos** dentro de un subespacio y presupuesto permitidos. Después se aplica el mismo algoritmo de transferencia que en los controles y se evalúa en tareas nuevas. Prioriza un par de 1.5B con una ventaja clara del donante y procedencia verificable. Compara contra receptor, donante, transferencia convencional, regreso a base si aplica, heurística, método numérico y controles aleatorios igualados. El perfil `maxxing` maximiza la habilidad objetivo y admite sacrificios secundarios. Reporta resultados negativos si los hay; no declares mejora o causalidad sin evidencia. **No añadas flotas, Project Mode, interfaz gráfica ni múltiples algoritmos hasta cerrar esta fase.**

---

## 11. Criterio de terminado

La v0.4 termina cuando podemos responder con artefactos y datos a estas preguntas:

1. ¿El donante elegido es realmente mejor que el receptor en la habilidad objetivo?
2. ¿Qué estímulos y observaciones llevaron a Claude a escoger esas regiones?
3. ¿Qué números del parche decidió Claude y cuáles calculó automáticamente el sistema?
4. ¿Cambió el comportamiento del receptor **antes** de la transferencia?
5. ¿El parche conserva su efecto **después** de la transferencia?
6. ¿Supera a una heurística y a un optimizador con la misma información y presupuesto?
7. ¿Supera la alternativa simple de usar el donante o volver hacia la base?
8. ¿Puede reproducirse, exportarse y revertirse el checkpoint final?

**Principio rector:** el MVP no debe probar que Claude tiene acceso a pesos; eso ya está demostrado. Debe probar —o refutar— que **Claude puede diseñar una cirugía informada que aporte valor funcional real**.
