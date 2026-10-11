"""Procedimiento común de las tres entregas; se incorpora en cada notebook.

No depende de resultados guardados: cada ejecución vuelve a ajustar los modelos.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.display import display, Markdown
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, precision_recall_curve, average_precision_score)

SEMILLA = 777
FUENTE = 'Fuente: OULAD, studentInfo.csv. Elaboración propia.'
NUMERICAS = ['num_of_prev_attempts', 'studied_credits']
CATEGORICAS = ['code_module', 'code_presentation', 'gender', 'region',
               'highest_education', 'imd_band', 'age_band', 'disability']
VARIABLES = NUMERICAS + CATEGORICAS
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda v: f'{v:.4f}')
plt.rcParams.update({'figure.figsize': (8, 5), 'figure.dpi': 100})


def cargar_datos():
    candidatos = [Path('Data/studentInfo.csv'), Path('../Data/studentInfo.csv')]
    ruta = next((p.resolve() for p in candidatos if p.is_file()), None)
    if ruta is None:
        raise FileNotFoundError('Ejecutar desde la raíz del repositorio o desde Codigo; falta Data/studentInfo.csv')
    datos = pd.read_csv(ruta)
    assert datos.shape[1] == 12
    assert set(datos.final_result.unique()) == {'Withdrawn', 'Pass', 'Fail', 'Distinction'}
    assert datos.id_student.notna().all()
    datos['abandono'] = datos.final_result.eq('Withdrawn').astype(int)
    print('Archivo local:', ruta.name)
    print('SHA256:', hashlib.sha256(ruta.read_bytes()).hexdigest())
    return datos


def particionar(datos):
    # Las fracciones de GroupShuffleSplit se refieren a estudiantes, no a filas.
    indices = np.arange(len(datos))
    grupos = datos.id_student.to_numpy()
    entrenamiento, temporal = next(GroupShuffleSplit(n_splits=1, test_size=.30,
        random_state=SEMILLA).split(indices, groups=grupos))
    validacion_local, prueba_local = next(GroupShuffleSplit(n_splits=1, test_size=.50,
        random_state=SEMILLA).split(temporal, groups=grupos[temporal]))
    partes = {'Entrenamiento': entrenamiento, 'Validación': temporal[validacion_local],
              'Prueba': temporal[prueba_local]}
    conjuntos = [set(grupos[idx]) for idx in partes.values()]
    assert all(conjuntos[i].isdisjoint(conjuntos[j]) for i in range(3) for j in range(i+1, 3))
    assert len(set(np.concatenate(list(partes.values())))) == len(datos)
    assert sum(map(len, partes.values())) == len(datos)
    tabla = pd.DataFrame([{'Conjunto': nombre, 'Inscripciones': len(idx),
        'Estudiantes': datos.iloc[idx].id_student.nunique(),
        'Inscripciones (%)': len(idx)/len(datos)*100,
        'Abandono (%)': datos.iloc[idx].abandono.mean()*100} for nombre, idx in partes.items()])
    assert all(datos.iloc[idx].abandono.nunique() == 2 for idx in partes.values())
    return partes, tabla


def preprocesador():
    # Categorías y mediana se aprenden SOLO con las filas del ajuste correspondiente.
    categorico = Pipeline([('imputar', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('codificar', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numericas', SimpleImputer(strategy='median'), NUMERICAS),
                              ('categoricas', categorico, CATEGORICAS)])


def preparar(datos, partes):
    X = datos[VARIABLES].copy()
    assert not {'id_student', 'final_result', 'abandono'}.intersection(X.columns)
    return {nombre: (X.iloc[idx], datos.iloc[idx].abandono) for nombre, idx in partes.items()}


def ajustar_logit(conjuntos, datos, partes):
    prep = preprocesador()
    X, y = conjuntos['Entrenamiento']
    z = prep.fit_transform(X)
    columnas = prep.get_feature_names_out()
    diseno = pd.DataFrame(sm.add_constant(z, has_constant='add'), index=X.index,
                         columns=['const'] + list(columnas))
    assert np.linalg.matrix_rank(diseno.to_numpy()) == diseno.shape[1], 'Diseño singular'
    modelo = sm.Logit(y, diseno).fit(maxiter=200, disp=False, cov_type='cluster',
        cov_kwds={'groups': datos.iloc[partes['Entrenamiento']].id_student.to_numpy()})
    assert modelo.mle_retvals['converged'], 'Logit no convergió'
    probabilidades = {}
    for nombre, (xp, _) in conjuntos.items():
        dp = pd.DataFrame(sm.add_constant(prep.transform(xp), has_constant='add'),
            index=xp.index, columns=diseno.columns)
        probabilidades[nombre] = np.asarray(modelo.predict(dp))
    intervalos = modelo.conf_int()
    coeficientes = pd.DataFrame({'Coeficiente': modelo.params, 'Odds ratio': np.exp(modelo.params),
        'OR inferior 95%': np.exp(intervalos[0]), 'OR superior 95%': np.exp(intervalos[1]),
        'p-value (cluster estudiante)': modelo.pvalues})
    referencias = dict(zip(CATEGORICAS, [c[0] for c in prep.named_transformers_['categoricas']
                                              .named_steps['codificar'].categories_]))
    return {'modelo': modelo, 'prep': prep, 'prob': probabilidades,
            'coeficientes': coeficientes, 'referencias': referencias}


def ajustar_arbol(conjuntos, datos, partes):
    X, y = conjuntos['Entrenamiento']
    grupos = datos.iloc[partes['Entrenamiento']].id_student.to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    pliegues = list(cv.split(X, y, grupos))
    for a, b in pliegues:
        assert set(grupos[a]).isdisjoint(set(grupos[b]))
    pipe = Pipeline([('preparacion', preprocesador()),
        ('modelo', DecisionTreeClassifier(random_state=SEMILLA))])
    grilla = {'modelo__max_depth': [3, 5, 8, 10],
              'modelo__min_samples_leaf': [20, 50],
              'modelo__min_samples_split': [2, 100]}
    # AUC es independiente del umbral; el punto de corte se decide después.
    busqueda = GridSearchCV(pipe, grilla, cv=pliegues, scoring='roc_auc', n_jobs=1,
                           error_score='raise', return_train_score=True)
    busqueda.fit(X, y)
    mejor = busqueda.best_estimator_
    prob = {n: mejor.predict_proba(x)[:, 1] for n, (x, _) in conjuntos.items()}
    nombres = mejor.named_steps['preparacion'].get_feature_names_out()
    importancias = pd.Series(mejor.named_steps['modelo'].feature_importances_, index=nombres,
                             name='Reducción de impureza').sort_values(ascending=False)
    return {'pipeline': mejor, 'modelo': mejor.named_steps['modelo'], 'prob': prob,
            'nombres': nombres, 'importancias': importancias, 'busqueda': busqueda}


def elegir_umbrales(conjuntos, logit, arbol):
    y = conjuntos['Validación'][1]
    candidatos = np.round(np.arange(.10, .601, .01), 2)
    tabla = pd.DataFrame([{'Umbral': u,
        'Precision': precision_score(y, logit['prob']['Validación'] >= u, zero_division=0),
        'Recall': recall_score(y, logit['prob']['Validación'] >= u),
        'F1': f1_score(y, logit['prob']['Validación'] >= u)} for u in candidatos])
    # En empates, se conserva el primer candidato (menor umbral).
    ul = float(tabla.loc[tabla.F1.idxmax(), 'Umbral'])
    fpr, tpr, cortes = roc_curve(y, arbol['prob']['Validación'])
    finitos = np.isfinite(cortes)
    youden = pd.DataFrame({'Umbral': cortes[finitos], 'Sensibilidad': tpr[finitos],
        'Especificidad': 1-fpr[finitos], 'Youden': (tpr-fpr)[finitos]})
    ua = float(youden.loc[youden.Youden.idxmax(), 'Umbral'])
    return {'Logit': ul, 'Árbol': ua}, tabla, youden


def metricas(y, prob, umbral, nombre):
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'Modelo': nombre, 'Umbral': umbral, 'Accuracy': accuracy_score(y, pred),
        'Precision': precision_score(y, pred, zero_division=0), 'Recall': recall_score(y, pred),
        'F1': f1_score(y, pred), 'AUC-ROC': roc_auc_score(y, prob),
        'VN': tn, 'FP': fp, 'FN': fn, 'VP': tp}


def evaluar(conjuntos, logit, arbol, umbrales):
    y = conjuntos['Prueba'][1]
    filas = []
    for nombre, objeto in [('Logit', logit), ('Árbol', arbol)]:
        for umbral, sufijo in [(.5, '0.50'), (umbrales[nombre], 'ajustado')]:
            filas.append(metricas(y, objeto['prob']['Prueba'], umbral, f'{nombre} {sufijo}'))
    base = DummyClassifier(strategy='most_frequent').fit(*conjuntos['Entrenamiento'])
    filas.append(metricas(y, base.predict_proba(conjuntos['Prueba'][0])[:, 1], .5, 'Mayoritaria'))
    return pd.DataFrame(filas).set_index('Modelo')


def mostrar_errores(conjuntos, objeto, umbral, nombre):
    y = conjuntos['Prueba'][1]
    pred = (objeto['prob']['Prueba'] >= umbral).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=[0, 1],
        display_labels=['No abandono', 'Abandono'], ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{nombre}: prueba, umbral {umbral:.3f}')
    ax.set_xlabel('Clase predicha'); ax.set_ylabel('Clase real')
    fig.text(.01, .01, FUENTE, fontsize=8); fig.tight_layout(rect=(0, .04, 1, 1)); plt.show()
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f'VP={tp}: abandonos detectados. FN={fn}: abandonos sin alerta y posible ayuda omitida.')
    print(f'FP={fp}: falsas alertas que consumen tutorías. VN={tn}: no abandono correctamente clasificado.')
    print(classification_report(y, pred, target_names=['No abandono', 'Abandono'], zero_division=0))


def mostrar_curvas(conjuntos, objetos):
    y = conjuntos['Prueba'][1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for nombre, objeto in objetos.items():
        prob = objeto['prob']['Prueba']
        fpr, tpr, _ = roc_curve(y, prob)
        p, r, _ = precision_recall_curve(y, prob)
        axes[0].plot(fpr, tpr, label=f'{nombre}: AUC={roc_auc_score(y, prob):.3f}')
        axes[1].plot(r, p, label=f'{nombre}: AP={average_precision_score(y, prob):.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[1].axhline(y.mean(), ls='--', color='grey', label='Prevalencia')
    axes[0].set(title='ROC en prueba', xlabel='Tasa de falsos positivos', ylabel='Recall')
    axes[1].set(title='Precision–Recall en prueba', xlabel='Recall', ylabel='Precision')
    for ax in axes: ax.legend()
    fig.text(.01, .01, FUENTE, fontsize=8); fig.tight_layout(rect=(0, .05, 1, 1)); plt.show()


def mostrar_importancias(logit, arbol):
    print('Referencias categóricas de Logit:', logit['referencias'])
    display(logit['coeficientes'])
    credito = logit['coeficientes'].loc['numericas__studied_credits']
    print(f"Por crédito adicional, las odds se multiplican por {credito['Odds ratio']:.4f}; "
          f"por 60 créditos, por {np.exp(60*credito['Coeficiente']):.3f}, manteniendo las demás variables.")
    display((arbol['importancias'].head(10)*100).to_frame('Importancia (%)'))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    co = logit['coeficientes'].drop(index='const').Coeficiente
    top = co.abs().nlargest(12).index
    co.loc[top].sort_values().plot.barh(ax=axes[0])
    axes[0].set(title='Coeficientes Logit (categorías y unidades originales)', xlabel='Log-odds')
    (arbol['importancias'].head(10).sort_values()*100).plot.barh(ax=axes[1])
    axes[1].set(title='Importancia del árbol', xlabel='Reducción total de impureza (%)')
    fig.text(.01, .01, FUENTE, fontsize=8); fig.tight_layout(rect=(0, .05, 1, 1)); plt.show()


def mostrar_arbol(arbol, conjuntos, datos, partes, umbral):
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(arbol['modelo'], max_depth=2, feature_names=arbol['nombres'],
        class_names=['No abandono', 'Abandono'], filled=True, proportion=True,
        rounded=True, fontsize=9, ax=ax)
    ax.set_title('Primeros dos niveles del árbol (el modelo completo continúa en (...))')
    fig.text(.01, .01, FUENTE, fontsize=9); fig.tight_layout(rect=(0, .04, 1, 1)); plt.show()
    # Elegir una fila de VALIDACIÓN, sin buscar ejemplos favorables en prueba.
    X = conjuntos['Validación'][0].iloc[[0]]
    z = arbol['pipeline'].named_steps['preparacion'].transform(X)
    display(X)
    ruta = arbol['modelo'].decision_path(z).indices
    t = arbol['modelo'].tree_
    print('Ruta completa de una inscripción real de validación:')
    for nodo in ruta:
        if t.children_left[nodo] == -1:
            prob = t.value[nodo, 0, 1]/t.value[nodo, 0].sum()
            print(f'Hoja {nodo}: {t.n_node_samples[nodo]} registros TRAIN, P(abandono)={prob:.4f}.')
        else:
            j = t.feature[nodo]; valor = z[0, j]; corte = t.threshold[nodo]
            signo = '<=' if valor <= corte else '>'
            print(f"Nodo {nodo}: {arbol['nombres'][j]} = {valor:g} {signo} {corte:.3f}; "
                  f"seguir {'izquierda' if valor <= corte else 'derecha'}.")
    p = arbol['pipeline'].predict_proba(X)[0, 1]
    print(f'Con umbral de validación {umbral:.4f}, la alerta final es {int(p >= umbral)}.')
    print('La clase mayoritaria del gráfico usa 0.50; la alerta operativa usa el umbral ajustado.')


def conclusiones(resultados):
    finales = resultados.loc[['Logit ajustado', 'Árbol ajustado']]
    mejor = finales.Recall.idxmax()
    fila = finales.loc[mejor]
    display(Markdown(f'Para priorizar la detección de abandono se prefiere **{mejor}** en esta partición: '
        f'Recall **{fila.Recall:.4f}**, F1 **{fila.F1:.4f}**, **{int(fila.FN)} falsos negativos** '
        f'y **{int(fila.FP)} falsas alertas**. La capacidad de tutorías debe evaluarse antes del uso real.'))
    for nombre in finales.index:
        f = finales.loc[nombre]
        print(f'{nombre}: meta Recall >= .75: {f.Recall >= .75}; meta F1 >= .70: {f.F1 >= .70}')
        base = resultados.loc[nombre.split()[0] + ' 0.50']
        print(f'Ajuste del umbral frente a .50: ΔRecall={f.Recall-base.Recall:+.4f}, '
              f'ΔF1={f.F1-base.F1:+.4f}, ΔFP={int(f.FP-base.FP):+d}.')
    print('Ajustar el umbral modifica la detección; no demuestra reducir la tasa real de abandono.')
