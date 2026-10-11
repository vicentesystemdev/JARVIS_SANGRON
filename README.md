# JARVIS_SANGRON

Predicción del riesgo de abandono estudiantil en educación superior. Proyecto de Minería de
Datos, Ingeniería de Sistemas, UNIFRANZ, semestre 2-2026; metodología CRISP-DM.

El objetivo es estimar la probabilidad de retiro de una inscripción y apoyar la priorización
académica de contactos y tutorías. La meta futura es reducir relativamente 10% la tasa histórica
(31,16% a 28,04%). Esta reducción de negocio todavía no se ha demostrado.

## Integrantes

Nombres documentados en `Codigo/01_entendimiento_negocio.ipynb`:

- Carla Valeria Encinas Cano
- Victor Asturizaga Plata
- Joel Herrera Pericon
- Joel Eduardo Condoreno Chura
- Vicente Oscar Claros Mamani

## Dataset y metodología

OULAD, `Data/studentInfo.csv`: 32.593 inscripciones, 12 variables originales, 28.785 estudiantes.
Cada fila representa estudiante–módulo–presentación. `abandono=1` corresponde a `Withdrawn`;
`Pass`, `Fail` y `Distinction` son clase 0. Fallar no equivale a retiro. Hay 10.156 abandonos,
22.437 no abandonos y 1.111 faltantes de `imd_band`. No se elimina ninguna inscripción.

El trabajo cubre comprensión del negocio, comprensión de datos, preparación y modelado de
CRISP-DM, con evaluación predictiva. La fase de despliegue institucional no se ha realizado.
Los datos de INE incluidos previamente sirven como contexto; no se unen a individuos de OULAD.

Se conserva `statsmodels.Logit` con intercepto, probabilidades, odds ratio e inferencia agrupada
por estudiante. El árbol es `DecisionTreeClassifier`, con búsqueda de hiperparámetros en un
Pipeline y CV de cinco pliegues de `StratifiedGroupKFold`, elegida por AUC-ROC.

La partición común se hace por estudiante, `GroupShuffleSplit`, semilla 777: 22.802 inscripciones
para entrenamiento, 4.933 para validación y 4.858 para prueba (aproximadamente 70/15/15).
No hay estudiantes compartidos entre conjuntos; el reparto anterior compartía 787 entre TRAIN
y TEST. Imputación y codificación se ajustan solo en entrenamiento, también dentro de cada
pliegue de CV. `final_result` nunca es predictor; `id_student` solo identifica grupos.
El punto de corte se elige en validación por F1 para Logit y por Youden para el árbol.

## Tres entregas ejecutadas

1. [Regresión logística](Codigo/02_regresion_logistica.ipynb).
2. [Árboles de decisión](Codigo/03_arbol_decision.ipynb).
3. [Evaluación grupal CRISP-DM, entrega 1](h3_03_evaluacion1_grupal_proyecto_crisp.ipynb),
   en la raíz y con las diez preguntas en el orden de la rúbrica.

`Codigo/respuestas_evaluacion.ipynb` conserva el avance, sincronizado con la entrega final.
Cada entrega incluye su código común y ajusta ambos modelos: no necesita ejecutar otro notebook
ni cargar resultados de entrenamientos previos. Las salidas y gráficos están guardados.

## Resultados verificados

Prueba: 4.858 inscripciones, 1.526 abandonos. Métricas de la clase positiva:

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Logit ajustado | 0.6097 | 0.4244 | 0.6809 | 0.5229 | 0.6884 |
| Árbol ajustado | 0.6157 | 0.4257 | 0.6402 | 0.5114 | 0.6649 |
| Mayoritaria | 0.6859 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

Logit usa umbral 0,3000: VN=1.923, FP=1.409, FN=487, VP=1.039.
El árbol usa umbral 0,3018868: VN=2.014, FP=1.318, FN=549, VP=977.
El árbol seleccionado tiene profundidad máxima 10, mínimo 50 por hoja y 2 para dividir.

Para minimizar abandonos omitidos, **Logit tiene mayor recall** y omite 62 abandonos menos,
a cambio de 91 falsas alertas adicionales en esta prueba. La institución necesita capacidad
para atenderlas. Ambos modelos ajustados tienen menor accuracy que la línea base mayoritaria,
pero esta no identifica ningún abandono. Las metas técnicas Recall >= 0,75 y F1 >= 0,70
no se alcanzaron; no se alteraron predicciones para cumplirlas.

La mejora medida es el efecto del umbral frente a 0,50: Logit aumenta recall de 0,1822 a
0,6809 y F1 de 0,2822 a 0,5229, con FP de 166 a 1.409. El árbol aumenta recall de 0,1855 a
0,6402 y F1 de 0,2819 a 0,5114, con FP de 199 a 1.318. No demuestra una reducción real del abandono.
Los cortes son políticas distintas; las entregas incluyen también la comparación homogénea a 0,50.
Las cifras proceden de [resultados calculados](Documentacion/resultados_verificados.json), generados
por la ejecución independiente y comparados entre los tres notebooks.

## Instalación y ejecución (Windows, Python 3.13)

Desde la raíz del repositorio, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name jarvis-entrega --display-name "Python 3 (JARVIS)"
.\.venv\Scripts\jupyter lab
```

Elegir el kernel **Python 3 (JARVIS)** y ejecutar **Restart Kernel and Run All Cells**.
Los notebooks pueden ejecutarse desde la raíz o desde `Codigo`; leen el dataset local.
Las versiones instaladas y verificadas en Windows con Python 3.13.0 están fijadas en
`requirements.txt`, incluidas las dependencias transitivas.

Para reproducir la verificación automática (entrena de nuevo, guarda las salidas y comprueba
la igualdad de resultados entre las tres entregas):

```powershell
.\.venv\Scripts\python Codigo/verificar_entregas.py
```

`Codigo/preparar_entregas.py` reconstruye las celdas desde las explicaciones del commit original
identificado en la auditoría y `Codigo/metodologia.py`; borra salidas de esos notebooks, por lo
que después hay que ejecutar el verificador. No es necesario para usar las entregas existentes.

## Estructura

- `Codigo/`: notebooks de comprensión del negocio, modelos y avance; procedimiento común,
  preparación de entregas y verificador.
- `Data/`: dataset OULAD, configuración del target y fuente complementaria de INE.
- `Documentacion/`: fuentes de datos, auditoría H3 y resultados calculados.
- `Presentaciones/`: materiales previos del grupo; para métricas H3 usar las entregas ejecutadas.
- Raíz: entrega CRISP-DM obligatoria, README y dependencias.

El archivo `Documentacion/JARVIS_SANGRON_proy_dm_f1_e3.pdf` tenía 0 bytes y se retiró.
No se encontró el documento verdadero en el proyecto. El PDF de `Presentaciones` es otro material.

## Límites y próximos pasos

Los datos describen Open University y presentaciones históricas; no acreditan desempeño futuro
ni transferencia a UNIFRANZ. Separar estudiantes resuelve dependencia entre conjuntos, pero
no sustituye una validación temporal. No se evaluó calibración ni un costo cuantitativo de FN/FP.
Las variables demográficas y el nivel de privación pueden requerir revisión de errores por grupo.
Las relaciones y las importancias no prueban causalidad.

Una siguiente iteración puede incorporar clics y primeras evaluaciones de OULAD con fecha de
corte explícita, seguida de validación en una cohorte futura. Ese experimento no se ejecutó.
La disponibilidad de la carga de créditos al momento de una alerta debe confirmarse.
La meta de negocio necesita evaluar intervenciones reales, por ejemplo con un diseño controlado.

## Referencias

- [OULAD en UCI](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset).
- [Kuzilek, Hlosta y Zdrahal (2017): Open University Learning Analytics dataset](https://doi.org/10.1038/sdata.2017.171).
- [Descarga oficial documentada por el grupo](https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip).
- [Repositorio y rúbrica del docente](https://github.com/ealaurel/MINERIA_DATOS_2026_2).
- [Auditoría metodológica y de ejecución](Documentacion/AUDITORIA_H3.md).

El usuario debe entregar en Moodle el enlace a este repositorio y los tres notebooks H3.
