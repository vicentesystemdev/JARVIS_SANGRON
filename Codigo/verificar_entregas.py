"""Ejecuta cada entrega en un kernel nuevo, guarda salidas y comprueba coherencia."""
from pathlib import Path
import json
import hashlib
import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
ENTREGAS = ['h3_03_evaluacion1_grupal_proyecto_crisp.ipynb',
            'Codigo/02_regresion_logistica.ipynb', 'Codigo/03_arbol_decision.ipynb']

def main():
    mediciones = []
    for archivo in ENTREGAS:
        ruta = RAIZ/archivo
        n = nbformat.read(ruta, as_version=4)
        evidencia = nbformat.v4.new_code_cell('''import json
evidencia = {
    'metricas': json.loads(resultados.to_json(orient='index')),
    'particiones': json.loads(resumen_particiones.to_json(orient='records')),
    'umbrales': umbrales,
    'hiperparametros': arbol['busqueda'].best_params_,
    'auc_cv': float(arbol['busqueda'].best_score_),
    'convergencia_logit': bool(logit['modelo'].mle_retvals['converged']),
    'sha256_dataset': hashlib.sha256((Path('Data/studentInfo.csv') if Path('Data/studentInfo.csv').exists() else Path('../Data/studentInfo.csv')).read_bytes()).hexdigest()
}
print('EVIDENCIA_JSON=' + json.dumps(evidencia, ensure_ascii=False))''')
        n.cells.append(evidencia)
        print('Ejecutando desde kernel limpio:', archivo, flush=True)
        NotebookClient(n, timeout=900, kernel_name='jarvis-entrega',
                       resources={'metadata':{'path':str(ruta.parent)}}).execute()
        salida = ''.join(o.get('text','') for o in n.cells[-1].outputs)
        valores = json.loads(salida.split('EVIDENCIA_JSON=')[1])
        mediciones.append(valores)
        n.cells.pop()
        nbformat.validate(n)
        for celda in n.cells:
            if celda.cell_type == 'code':
                assert celda.execution_count is not None
                assert all(o.output_type != 'error' for o in celda.outputs)
        nbformat.write(n, ruta)
        print('OK:', archivo, flush=True)
    assert mediciones[0] == mediciones[1] == mediciones[2], 'Resultados diferentes entre entregas'
    # El avance conserva también salidas, con el mismo código independiente.
    nbformat.write(nbformat.read(RAIZ/ENTREGAS[0], as_version=4), RAIZ/'Codigo/respuestas_evaluacion.ipynb')
    (RAIZ/'Documentacion/resultados_verificados.json').write_text(
        json.dumps(mediciones[0], indent=2, ensure_ascii=False), encoding='utf-8')
    print('COHERENCIA: métricas, particiones, umbrales y parámetros idénticos en las tres entregas.')

if __name__ == '__main__':
    main()
