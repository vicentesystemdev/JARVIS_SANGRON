"""Actualiza entregas a partir de sus explicaciones existentes y el procedimiento común."""
from pathlib import Path
import copy
import subprocess
import nbformat as nb

RAIZ = Path(__file__).resolve().parents[1]
originales = {n: nb.reads(subprocess.check_output(['git', 'show', f'2af1f14cc4036683302337b2ea937c7e62e37f5a:Codigo/{n}.ipynb'],
              cwd=RAIZ).decode('utf-8'), as_version=4)
              for n in ['02_regresion_logistica', '03_arbol_decision', 'respuestas_evaluacion']}
comun = (RAIZ/'Codigo/metodologia.py').read_text(encoding='utf-8')
md = nb.v4.new_markdown_cell
code = nb.v4.new_code_cell
integrantes = 'Carla Valeria Encinas Cano; Victor Asturizaga Plata; Joel Herrera Pericon; Joel Eduardo Condoreno Chura; Vicente Oscar Claros Mamani.'
nota = '''## Corrección metodológica de la entrega
Se conserva el problema, el target, las explicaciones de negocio y los modelos desarrollados por el grupo.
Se sustituyen las particiones por filas y las cifras manuales por una única partición **por estudiante**
(semilla 777), aproximadamente 70/15/15. Se estima el desempeño para **estudiantes no vistos** en las
presentaciones históricas disponibles. No es una evaluación temporal de un curso futuro.

`final_result` solo construye el target; `id_student` solo forma grupos. La preparación se ajusta
exclusivamente en entrenamiento. El código común está incluido aquí: este notebook no necesita
ejecutar otro notebook ni leer métricas anteriores. Se ejecuta desde la raíz o `Codigo`.
Fuente de todas las tablas y gráficos: OULAD, `studentInfo.csv`; elaboración propia.
'''
datos = '''df = cargar_datos()
partes, resumen_particiones = particionar(df)
conjuntos = preparar(df, partes)
display(df.head())
display(pd.DataFrame({'Cantidad': df.abandono.value_counts().sort_index(),
                     'Porcentaje (%)': df.abandono.value_counts(normalize=True).sort_index()*100}))
print('Dimensiones originales:', (len(df), df.shape[1]-1))
print('Estudiantes únicos:', df.id_student.nunique())
display(resumen_particiones)
print('Intersecciones de estudiantes entre conjuntos: 0 (verificado con assertions).')
# Auditoría del procedimiento anterior: estratificar filas no separa estudiantes.
a, resto = train_test_split(np.arange(len(df)), test_size=.30, random_state=777, stratify=df.abandono)
v, b = train_test_split(resto, test_size=.50, random_state=777, stratify=df.iloc[resto].abandono)
print('Estudiantes compartidos TRAIN/TEST en el método anterior:',
      len(set(df.iloc[a].id_student) & set(df.iloc[b].id_student)))
'''
calidad = '''display(pd.DataFrame({'Tipo': df.dtypes.astype(str), 'Faltantes': df.isna().sum(),
                      'Valores distintos': df.nunique()}))
print('Duplicados exactos:', df.drop(columns='abandono').duplicated().sum())
print('Claves inscripción duplicadas:', df.duplicated(['id_student','code_module','code_presentation']).sum())
print('Faltantes imd_band (%):', df.imd_band.isna().mean()*100)
for variable in NUMERICAS:
    q1, q3 = df[variable].quantile([.25,.75])
    inferior, superior = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
    atipicos = ((df[variable]<inferior)|(df[variable]>superior)).sum()
    print(variable, 'rango:', df[variable].min(), df[variable].max(),
          'límites IQR:', inferior, superior, 'posibles atípicos:', atipicos)
    print('Negativos:', (df[variable]<0).sum(), 'No enteros:', (df[variable]%1!=0).sum())
for variable in CATEGORICAS:
    print(variable, sorted(df[variable].dropna().unique()))
print('Etiquetas imd_band sin %:', df.imd_band.dropna()[~df.imd_band.dropna().str.endswith('%')].value_counts().to_dict())
'''
preparacion = '''### Preparación aplicada y límites
Se conservan todas las inscripciones: repetir un estudiante no equivale a duplicar una inscripción.
`imd_band` faltante se representa con **Unknown**; la mediana numérica y las categorías del
`OneHotEncoder(drop='first')` se ajustan con entrenamiento. La etiqueta `10-20` se conserva como
categoría porque es una inconsistencia de formato, sin ambigüedad con otra banda registrada.
Los valores altos de créditos y de intentos previos se conservan: el IQR señala rareza, no errores.
Los controles y sus cantidades se muestran arriba, sin afirmar que prueban la veracidad de cada registro.

No se agregan datos de INE a las filas de OULAD: son poblaciones distintas. Se usan los diez
predictores originales que describen contexto, demografía y matrícula, sin variables posteriores al
resultado. Su disponibilidad al momento de aplicar una alerta debe verificarse institucionalmente.
`studied_credits` recoge la carga de créditos del estudiante, no necesariamente solo la del módulo.
'''
eda = '''# Hallazgo conservado del análisis original: abandono según carga de créditos.
# Este gráfico descriptivo usa todo el dataset; no se usa para seleccionar variables ni modelos.
rango = pd.cut(df.studied_credits, bins=[0,60,120,180,np.inf], include_lowest=True,
               labels=['Hasta 60','61-120','121-180','Más de 180'])
tabla_creditos = df.groupby(rango, observed=True).abandono.agg(['size','sum','mean'])
tabla_creditos.columns = ['Inscripciones','Abandonos','Tasa']
display(tabla_creditos.assign(**{'Tasa (%)':tabla_creditos.Tasa*100}).drop(columns='Tasa'))
fig, ax = plt.subplots()
barras = ax.bar(tabla_creditos.index.astype(str), tabla_creditos.Tasa*100)
ax.bar_label(barras, fmt='%.2f %%')
ax.axhline(df.abandono.mean()*100, ls='--', color='grey', label='Tasa histórica general')
ax.set(title='Abandono según carga de créditos', xlabel='Créditos del estudiante (rango)', ylabel='Abandono (%)')
ax.legend(); fig.text(.01,.01,FUENTE,fontsize=8)
fig.tight_layout(rect=(0,.05,1,1)); plt.show()
print(f'Diferencia entre extremos: {(tabla_creditos.Tasa.iloc[-1]-tabla_creditos.Tasa.iloc[0])*100:.2f} puntos porcentuales.')
print('¿Las tasas crecen en todos los rangos?:', tabla_creditos.Tasa.is_monotonic_increasing)
'''
ajustes = '''logit = ajustar_logit(conjuntos, df, partes)
arbol = ajustar_arbol(conjuntos, df, partes)
umbrales, tabla_validacion, tabla_youden = elegir_umbrales(conjuntos, logit, arbol)
print('Logit convergió:', logit['modelo'].mle_retvals['converged'])
print('Columnas transformadas:', len(arbol['nombres']))
print('Mejores hiperparámetros:', arbol['busqueda'].best_params_)
print('AUC medio CV (solo TRAIN):', arbol['busqueda'].best_score_)
print('Umbrales elegidos solo con validación:', umbrales)
'''
evaluacion = '''resultados = evaluar(conjuntos, logit, arbol, umbrales)
display(resultados)
mostrar_errores(conjuntos, logit, umbrales['Logit'], 'Regresión logística')
mostrar_errores(conjuntos, arbol, umbrales['Árbol'], 'Árbol de decisión')
mostrar_curvas(conjuntos, {'Logit':logit, 'Árbol':arbol})
conclusiones(resultados)
'''
limites = '''### Límites y mejora siguiente
La reducción del umbral es una mejora técnica medible frente a 0,50; las diferencias de Recall, F1
y falsas alertas se calculan arriba. **No se demuestra reducir el abandono real**, ni se alcanza una
meta por haber cambiado las reglas de evaluación. Las cifras históricas de otros repartos no son
comparables directamente con esta prueba por estudiantes.

Los falsos negativos omiten apoyo a personas que finalmente abandonan; los falsos positivos
consumen contactos y tutorías. No se conoce un costo monetario que permita optimizar esa decisión.
Youden equilibra sensibilidad y especificidad, no minimiza específicamente el costo de FN; F1
equilibra precisión y recall. La elección de un umbral con una cuota de tutorías o un costo explícito
requiere una nueva validación, sin volver a ajustar sobre esta prueba.

La importancia por impureza puede favorecer variables con más puntos de corte. Los coeficientes
no se comparan como ranking absoluto cuando las unidades son distintas. Sus asociaciones no son
causales. Logit usa errores estándar agrupados por estudiante para considerar las inscripciones
dependientes dentro de TRAIN. Las probabilidades no se han validado con un estudio de calibración.

Como trabajo futuro, agregar actividad de las primeras semanas de `studentVle` y evaluaciones
tempranas de `studentAssessment`, fijando una fecha de corte, y evaluar en una cohorte futura
con estudiantes separados. Esa mejora **todavía no fue ejecutada**. OULAD corresponde a Open
University; extrapolar a UNIFRANZ requiere datos locales y validación, así como revisar diferencias
de error entre grupos y confirmar la disponibilidad temporal de todos los predictores.
'''
referencias = '''## Referencias y reproducción
- [OULAD, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset)
- [Kuzilek, Hlosta y Zdrahal (2017), artículo del dataset](https://doi.org/10.1038/sdata.2017.171)
- [Repositorio del docente](https://github.com/ealaurel/MINERIA_DATOS_2026_2): `h3_01_Regresion_Logistica.ipynb`,
  `h3_02_Arboles_de_decisión_clasificacion.ipynb`, `h3_03_ejemplo_entrega_tarea.ipynb`
  y `h3_03_evaluacion1_grupal_proyecto_crisp_dm copy.md`.
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

Instalar `requirements.txt`, abrir Jupyter y ejecutar **Restart Kernel and Run All Cells**.
El código de `Codigo/metodologia.py` se incorpora en cada entrega para que ninguna dependa de
variables o resultados de otro notebook. Los resultados se obtienen del CSV local con semilla 777.
'''

def guardar(celdas, destino, plantilla):
    documento = nb.v4.new_notebook(cells=celdas, metadata=copy.deepcopy(plantilla.metadata))
    documento.metadata.kernelspec = {'display_name':'Python 3 (JARVIS)', 'language':'python', 'name':'jarvis-entrega'}
    nb.validate(documento)
    nb.write(documento, destino)

# Se aprovechan directamente los enunciados y las respuestas de negocio del avance.
viejo = originales['respuestas_evaluacion']
def anterior(i): return copy.deepcopy(viejo.cells[i])
intro = anterior(0)
intro.source = intro.source.replace('`../Data/studentInfo.csv`', '`Data/studentInfo.csv` (o `../Data/studentInfo.csv` desde `Codigo`)')
intro.source += '\n\n**Integrantes:** '+integrantes
preguntas = [intro, md(nota), code(comun), code('df = cargar_datos()')]
preguntas += [anterior(2), anterior(3), anterior(4), anterior(5), anterior(6)]
preguntas += [md('''### Pregunta 4. ¿Qué problemas de calidad encontraron y qué preparación aplicaron?
La tabla y los controles siguientes calculan faltantes, duplicados, claves repetidas, formatos y
posibles valores atípicos. Los tratamientos se justifican debajo; no se elimina ninguna inscripción.'''),
             code(calidad), md(preparacion)]
preguntas += [md('''### Pregunta 5. ¿Qué resultado interesante encontraron y qué gráfico lo demuestra?
Se conserva el hallazgo sobre la carga de créditos del análisis original. El gráfico calcula las tasas
y el tamaño de cada grupo. Las diferencias sugieren priorizar la revisión de cargas académicas;
son asociaciones descriptivas, sin demostrar que los créditos causen abandono ni que una tutoría
reduzca esa tasa. Los extremos con pocas inscripciones pueden ser menos estables.'''), code(eda)]
preguntas += [anterior(11), anterior(12)]
preguntas += [md('''### Pregunta 7. ¿Qué proporción usaron para entrenamiento, validación y prueba, y cómo hicieron la separación?
Se usa `GroupShuffleSplit` en dos pasos (70/30 de estudiantes; mitad del 30% para cada conjunto),
semilla 777. Las proporciones por **inscripciones** se calculan abajo y pueden variar ligeramente.
No se estratifica exactamente; se comprueba la prevalencia obtenida. Esto corresponde a predecir
inscripciones de estudiantes no vistos. Un reparto por filas puede compartir personas y patrones
entre entrenamiento y prueba incluso excluyendo su identificador; es riesgo de dependencia, no
prueba de que toda métrica previa estuviera inflada.

La misma partición se usa para ambos modelos. Preparación y búsqueda usan TRAIN; los cinco pliegues
de CV del árbol también separan estudiantes (`StratifiedGroupKFold`). Se elige por AUC, independiente
del umbral. VALIDACIÓN elige F1 de Logit y Youden del árbol; PRUEBA solo mide configuraciones
predefinidas. No se vuelven a entrenar los modelos con validación ni se seleccionan cortes con prueba.'''),
              code(datos.replace('df = cargar_datos()\n',''))]
preguntas += [md('''### Pregunta 8. ¿Cuáles son las variables más importantes y cómo determinaron su importancia?
Logit conserva la máxima verosimilitud de `statsmodels`, con intercepto y una categoría de referencia
por variable para evitar colinealidad. Se comprueba rango completo y convergencia. El árbol usa una
grilla de profundidades 3, 5, 8 y 10; mínimo por hoja 20 o 50; mínimo para dividir 2 o 100, con cinco
pliegues por grupos. Las cifras y parámetros definitivos se calculan aquí.

Los coeficientes y odds ratio representan asociaciones condicionadas a los demás predictores.
La importancia del árbol suma la reducción de impureza. La interpretación de créditos y las
categorías de referencia se imprimen debajo; ninguna de estas medidas establece causalidad.'''),
              code(ajustes), code('mostrar_importancias(logit, arbol)')]
preguntas += [md('''### Pregunta 9. ¿Qué desempeño obtuvo el modelo en el conjunto de prueba?
Se muestran las cinco métricas solicitadas y las matrices, calculadas para los dos modelos y una
línea base mayoritaria entrenada en TRAIN. Se compara el umbral 0,50 con el seleccionado en
VALIDACIÓN: es una comparación reproducible de la mejora técnica y de su costo en falsas alertas.
La prioridad son los falsos negativos. El modelo de mayor recall es preferible bajo esa prioridad,
con la condición de que la institución pueda atender sus alertas. La reducción de abandono propuesta
en negocio necesita una intervención real; no puede inferirse de estas métricas predictivas.'''),
              code('display(tabla_validacion.loc[tabla_validacion.F1.idxmax()].to_frame("Selección F1 en validación"))\ndisplay(tabla_youden.loc[tabla_youden.Youden.idxmax()].to_frame("Selección Youden en validación"))'),
              code(evaluacion), md(limites)]
preguntas += [md('''### Pregunta 10. ¿Qué visualización permite entender cómo toma decisiones el modelo?
El gráfico muestra dos niveles iniciales y la celda imprime **una ruta completa real de validación**,
con todos sus valores, condiciones, hoja y decisión final. Izquierda significa `<=`; derecha, `>`.
Para una dummy, `<=0.5` indica ausencia de esa categoría. `samples` cuenta inscripciones TRAIN;
`value` muestra proporciones [no abandono, abandono]. La profundidad visual se recorta por legibilidad,
pero la ruta impresa llega a la hoja real. La clase del dibujo usa mayoría (0,50), mientras que la
alerta del proyecto usa Youden: pueden diferir y la celda muestra el criterio correcto.'''),
              code("mostrar_arbol(arbol, conjuntos, df, partes, umbrales['Árbol'])"), md(referencias)]
guardar(preguntas, RAIZ/'h3_03_evaluacion1_grupal_proyecto_crisp.ipynb', viejo)
# El avance sigue en su ubicación, actualizado para no conservar resultados finales contradictorios.
guardar(copy.deepcopy(preguntas), RAIZ/'Codigo/respuestas_evaluacion.ipynb', viejo)

log = originales['02_regresion_logistica']
lg = [md('# 02 · Regresión Logística — Predicción de abandono académico\n\n**Proyecto:** JARVIS_SANGRON. **Dataset:** OULAD. **Clasificación binaria supervisada.**\n\nEstimar la probabilidad de abandono para priorizar seguimiento académico.'),
      md(nota), copy.deepcopy(log.cells[1]), code(comun),
      copy.deepcopy(log.cells[5]), code(datos), copy.deepcopy(log.cells[8]),
      copy.deepcopy(log.cells[11]), md('## Calidad y preparación'), code(calidad), md(preparacion),
      md('## Análisis exploratorio conservado'), code(eda),
      md('''## Primera iteración — modelo básico (evaluado en validación)
Se conserva el baseline interpretable con `num_of_prev_attempts` y `studied_credits`.
La iteración y el modelo ampliado usan el mismo entrenamiento y validación por estudiantes;
no se emplea prueba para explorar variables ni umbrales.'''),
      code('''Xtr, ytr = conjuntos['Entrenamiento']
Xv, yv = conjuntos['Validación']
pre_basico = SimpleImputer(strategy='median')
zb = sm.add_constant(pre_basico.fit_transform(Xtr[NUMERICAS]), has_constant='add')
basico = sm.Logit(ytr, zb).fit(disp=False, maxiter=200)
assert basico.mle_retvals['converged']
pv_basico = np.asarray(basico.predict(sm.add_constant(pre_basico.transform(Xv[NUMERICAS]), has_constant='add')))
display(pd.DataFrame([metricas(yv,pv_basico,.5,'Básico: validación')]).set_index('Modelo'))'''),
      copy.deepcopy(log.cells[16]), md('## Segunda iteración — incorporación de más variables'),
      md('''Se incorporan los diez predictores originales. Se conserva `statsmodels.Logit`: proporciona
coeficientes, odds ratio e intervalos. La preparación equivalente a un Pipeline se ajusta en TRAIN
y se reutiliza sin reajustar. La inferencia usa covarianza agrupada por estudiante, dado que TRAIN
contiene varias inscripciones por persona. Se comprueban rango completo y convergencia.'''),
      code('''logit = ajustar_logit(conjuntos, df, partes)
print(logit['modelo'].summary())
display(logit['coeficientes'])
print('Categorías de referencia:',logit['referencias'])
display(pd.DataFrame([metricas(yv,pv_basico,.5,'Básico: validación'),
    metricas(yv,logit['prob']['Validación'],.5,'Ampliado: validación')]).set_index('Modelo'))
credito = logit['coeficientes'].loc['numericas__studied_credits']
print(f"OR por crédito: {credito['Odds ratio']:.4f}; OR por 60 créditos: {np.exp(60*credito['Coeficiente']):.4f}")
co = logit['coeficientes'].drop(index='const').Coeficiente
co.loc[co.abs().nlargest(12).index].sort_values().plot.barh()
plt.title('Coeficientes de Logit (log-odds, unidades originales)')
plt.xlabel('Coeficiente'); plt.tight_layout(); plt.show()'''),
      copy.deepcopy(log.cells[27]), copy.deepcopy(log.cells[32]),
      md('Toda exploración de umbrales siguiente usa **VALIDACIÓN**, y la preparación usa únicamente TRAIN.'),
      code('''arbol = ajustar_arbol(conjuntos, df, partes)
umbrales, tabla_validacion, tabla_youden = elegir_umbrales(conjuntos, logit, arbol)
display(tabla_validacion)
tabla_validacion.set_index('Umbral')[['Precision','Recall','F1']].plot()
plt.axvline(umbrales['Logit'],ls='--',color='black',label=f"Elegido: {umbrales['Logit']:.2f}")
plt.title('Efecto del umbral en validación'); plt.ylabel('Métrica (0 a 1)'); plt.legend(); plt.tight_layout(); plt.show()'''),
      copy.deepcopy(log.cells[43]), code(evaluacion),
      md('## Probabilidades y clasificación de inscripciones reales'),
      code('''ejemplos = conjuntos['Prueba'][0].head(8).copy()
ejemplos['Probabilidad abandono'] = logit['prob']['Prueba'][:8]
ejemplos['Alerta'] = (ejemplos['Probabilidad abandono'] >= umbrales['Logit']).astype(int)
ejemplos['Clase real'] = conjuntos['Prueba'][1].head(8)
display(ejemplos)'''), md(limites), md(referencias)]
guardar(lg, RAIZ/'Codigo/02_regresion_logistica.ipynb', log)

tr = originales['03_arbol_decision']
ar = [copy.deepcopy(tr.cells[0]), md(nota), copy.deepcopy(tr.cells[1]), code(comun),
      copy.deepcopy(tr.cells[4]), code(datos), copy.deepcopy(tr.cells[7]),
      md('## Calidad y preparación'), code(calidad), md(preparacion),
      copy.deepcopy(tr.cells[12]),
      md('''## Búsqueda de hiperparámetros con GridSearchCV
Se buscan `max_depth` en [3,5,8,10], `min_samples_leaf` en [20,50] y `min_samples_split` en
[2,100]. Todos son válidos; la restricción de hojas y profundidad limita sobreajuste y facilita
la defensa. `criterion='gini'` y `class_weight=None`. Se maximiza AUC media de cinco pliegues de
`StratifiedGroupKFold`, usando solo TRAIN y estudiantes separados. Cada pliegue ajusta su propio
preprocesamiento dentro del Pipeline. Se usa `error_score='raise'` para no ocultar fallos.
La validación cruzada estratifica aproximadamente; no garantiza prevalencias idénticas por grupos.
Se informa también AUC de entrenamiento para observar la diferencia de generalización.'''),
      code(ajustes), code('''cv_resultados = pd.DataFrame(arbol['busqueda'].cv_results_)
display(cv_resultados[['params','mean_train_score','mean_test_score','std_test_score','rank_test_score']].sort_values('rank_test_score'))
print('Profundidad real:', arbol['modelo'].get_depth(), 'Hojas:',arbol['modelo'].get_n_leaves())'''),
      copy.deepcopy(tr.cells[22]), code('mostrar_importancias(logit, arbol)'),
      md(copy.deepcopy(tr.cells[25]).source.replace('82 créditos matriculados', 'el corte de créditos indicado en el nodo')),
      md('La ruta siguiente usa una inscripción de validación y continúa hasta su hoja; el dibujo muestra los primeros dos niveles.'),
      code("mostrar_arbol(arbol, conjuntos, df, partes, umbrales['Árbol'])"),
      md('''## Selección del umbral mediante Youden en validación
Se calcula $J=Sensibilidad+Especificidad-1=TPR-FPR$ y se elige el máximo entre cortes finitos.
En empates se conserva el primero de `roc_curve`. Youden pondera ambas tasas por igual y puede
producir muchas falsas alertas. El umbral se fija antes de evaluar PRUEBA. F1 de Logit y Youden
del árbol son políticas distintas; la tabla a 0,50 ofrece una comparación con un mismo corte.'''),
      code('''display(tabla_youden)
display(tabla_youden.loc[tabla_youden.Youden.idxmax()].to_frame('Máximo en validación'))
tabla_youden.sort_values('Umbral').plot(x='Umbral',y=['Sensibilidad','Especificidad','Youden'])
plt.title('Youden y sus componentes en validación'); plt.ylabel('Tasa / índice'); plt.tight_layout(); plt.show()'''),
      copy.deepcopy(tr.cells[33]), copy.deepcopy(tr.cells[35]),
      md('## Evaluación verificable y comparación con Logit y línea base'), code(evaluacion),
      md(limites), md(referencias)]
guardar(ar, RAIZ/'Codigo/03_arbol_decision.ipynb', tr)
print('Cuatro notebooks preparados; tres entregas y el avance sincronizado.')
