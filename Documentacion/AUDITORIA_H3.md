# Auditoría de las tres entregas H3

Fecha: 11 de octubre de 2026. Proyecto: JARVIS_SANGRON, UNIFRANZ, Minería de Datos.

## Alcance y preservación del trabajo

Se revisaron los cuatro notebooks originales y la estructura del repositorio antes de editar.
Estado inicial: rama `main`, commit `2af1f14cc4036683302337b2ea937c7e62e37f5a`;
único cambio local en `Codigo/respuestas_evaluacion.ipynb`: versión del kernel de 3.12.3 a
3.14.7, sin cambios de contenido. La versión del entorno ahora se registra con su ejecución real.
Se conservaron el target, el planteamiento de negocio, la meta tentativa del 10%, el análisis de
créditos, los integrantes documentados y las explicaciones conceptuales correctas. El avance
`respuestas_evaluacion.ipynb` permanece, sincronizado con la entrega final y sus salidas.
El notebook de comprensión del negocio, el dataset y las presentaciones existentes se conservan.

## Referencias del docente revisadas

Se consultó una copia de lectura del [repositorio del docente](https://github.com/ealaurel/MINERIA_DATOS_2026_2),
commit `8e6c0ed934f50ee0e95cb0540075eaf13c67a5b2`:

- `h3_03_evaluacion1_grupal_proyecto_crisp_dm copy.md`: diez preguntas, evidencia calculada,
  claridad, métricas y reproducibilidad; se respondió en el mismo orden.
- `h3_03_ejemplo_entrega_tarea.ipynb`: formato de identificación y respuestas breves del proyecto.
- `h3_01_Regresion_Logistica.ipynb`: negocio, target, predictores, separación, `statsmodels.Logit`,
  coeficientes, probabilidades y métricas de clasificación.
- `h3_02_Arboles_de_decisión_clasificacion.ipynb`: `DecisionTreeClassifier`, `GridSearchCV`,
  probabilidades, Youden y matrices. Se conservaron los conceptos, adaptando la metodología:
  cortes con validación en vez de prueba y grilla con valores válidos (`min_samples_split >= 2`).

No se modificó el repositorio del docente ni se sustituyó OULAD por los ejemplos de clase.

## Hallazgos y correcciones

1. El dataset tiene 32.593 inscripciones y 28.785 estudiantes. El antiguo reparto estratificado
   70/15/15 compartía **787 estudiantes entre TRAIN y TEST**, comprobado al reconstruir sus índices.
   Aunque se excluía `id_student`, las observaciones de una persona podían quedar en ambos conjuntos.
   No se afirma que se haya demostrado una cantidad específica de inflación de las métricas.
2. Se adoptó `GroupShuffleSplit`, semilla 777, con estudiantes disjuntos en las tres particiones.
   Se obtienen 22.802 / 4.933 / 4.858 inscripciones (69,96% / 15,14% / 14,91%), con prevalencias
   de abandono 31,16% / 30,93% / 31,41%. Se busca generalización a estudiantes no vistos dentro
   de las presentaciones históricas, sin afirmar generalización temporal.
3. Se sustituyó codificar el dataset completo por `ColumnTransformer`, imputación y one-hot
   ajustados en TRAIN. En la CV del árbol se aprende la preparación de nuevo dentro de cada pliegue.
   `final_result` solo define `abandono`; `id_student` solo agrupa. Ninguno entra como predictor.
4. La regresión exploraba cortes sobre TEST de una iteración previa. Ahora las iteraciones y
   los cortes se exploran en validación. Se conservó `statsmodels.Logit`, con intercepto, diseño
   de rango completo, comprobación de convergencia y covarianza agrupada por estudiante.
5. La evaluación fijaba manualmente hiperparámetros y copiaba métricas de Logit. Ahora cada notebook
   ajusta ambos modelos desde cero; CV de cinco pliegues por estudiantes elige el árbol por AUC;
   validación selecciona el umbral F1 de Logit y el máximo Youden del árbol.
6. Se reportan métricas y errores computados, frente al corte 0,50 y una línea base mayoritaria.
   La mejora observada de recall/F1 al cambiar el corte se acompaña del aumento de falsas alertas.
   Agregar actividad temprana sigue siendo una propuesta, no un experimento realizado.
7. Los gráficos muestran datos calculados y una ruta completa real de validación hasta la hoja.
   Se distingue la clase mayoritaria del dibujo de la alerta determinada por Youden.
8. Calidad: 1.111 faltantes en `imd_band`, cero duplicados exactos y cero claves de inscripción
   duplicadas; 350 cargas de créditos sobre el límite IQR de 210. Se conservan filas y atípicos;
   `10-20` es una etiqueta sin `%`, mantenida como categoría. El IQR de intentos previos tiene
   rango intercuartílico cero; señalar 4.172 valores positivos no justifica eliminarlos.
9. La ejecución paralela inicial de GridSearchCV causó un error del recolector de archivos temporales
   de joblib en Windows fuera de las celdas. Se cambió a `n_jobs=1` y se ejecutaron nuevamente
   las tres entregas. No se ocultaron excepciones del ajuste (`error_score='raise'`).

## PDF y material previo

`Documentacion/JARVIS_SANGRON_proy_dm_f1_e3.pdf` tenía **0 bytes**; no se encontró una versión
real de ese documento dentro del proyecto. Se retiró el archivo vacío. El PDF de presentación
en `Presentaciones/` es otro archivo y se conserva; no se lo renombró como documento de proyecto.
Esta auditoría y los notebooks son evidencia nueva, no un reemplazo inventado del PDF ausente.

## Verificación reproducible

`Codigo/verificar_entregas.py` ejecuta las tres entregas en kernels nuevos, desde la ubicación
de cada archivo; valida nbformat/JSON, ejecución de todas las celdas sin salidas `error`,
convergencia y coincidencia exacta de métricas, particiones, parámetros y umbrales entre notebooks.
Dentro de cada notebook se comprueban exclusión de predictores prohibidos, disjunción entre
estudiantes, cobertura de filas, rango del diseño Logit, probabilidades finitas y clases presentes.
Se inspeccionaron visualmente el árbol y el gráfico de coeficientes/importancias.

Los resultados verificables están en `resultados_verificados.json` y en las salidas de las entregas;
el README presenta las métricas obtenidas. El SHA256 del dataset permite identificar la copia utilizada.

## Límites y acciones futuras

Ningún modelo alcanza simultáneamente Recall >= 0,75 y F1 >= 0,70. La meta de negocio (reducción
relativa del abandono de 10%) no fue evaluada mediante una intervención. No se conoce el costo
monetario de FN/FP ni la cuota de tutorías. No se hizo validación temporal, calibración ni
validación con estudiantes de UNIFRANZ. Las asociaciones no son evidencia causal.

Para Moodle deben usarse las tres entregas H3 ejecutadas que enlaza el README. Los materiales
de fases anteriores no sustituyen las métricas finales. El usuario subirá el enlace a Moodle.
Recuperar el PDF original solo es necesario si se requiere ese documento adicional.
