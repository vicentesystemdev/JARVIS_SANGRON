PAQUETE DE FUENTES - PROYECTO DE DESERCIÓN ESTUDIANTIL
Grupo: Jarvis
Fecha de preparación: 2026-08-28

1) DATASET PRINCIPAL: OULAD
Fuente oficial: The Open University
Descarga ZIP oficial:
https://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip

Espejo académico UCI Machine Learning Repository:
https://archive.ics.uci.edu/static/public/349/open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip

Archivo principal para el Project Charter (32.593 registros, 12 variables):
https://raw.githubusercontent.com/marloft/MachineLearning/refs/heads/master/Documents/ML/PhD/Datasets/Open%20University%20Learning%20Analytics%20Dataset%20-%20OULAD/studentInfo.csv

Target recomendado:
- abandono = 1 cuando final_result = Withdrawn
- abandono = 0 cuando final_result = Pass, Distinction o Fail

IMPORTANTE:
- final_result se utiliza para CONSTRUIR la etiqueta abandono.
- final_result NO debe usarse como predictor.
- date_unregistration tampoco debe usarse como predictor temprano si revela el retiro.

2) FUENTE COMPLEMENTARIA: ENCUESTA DE HOGARES 2025 - INE BOLIVIA
Catálogo oficial:
https://anda.ine.gob.bo/index.php/catalog/256

Acceso a microdatos:
https://anda.ine.gob.bo/index.php/catalog/256/get-microdata

Metadatos JSON:
https://anda.ine.gob.bo/index.php/metadata/export/256/json

Diccionario EH2025_Persona:
https://anda.ine.gob.bo/index.php/catalog/256/data-dictionary/F27?file_name=EH2025_Persona

IMPORTANTE:
El INE indica que la descarga de microdatos requiere registrar una cuenta
con usuario y correo electrónico. El catálogo y los metadatos son públicos,
pero el archivo de microdatos no debe ser reconstruido desde las páginas de
metadatos porque estas no contienen las 39.485 observaciones completas.

Uso recomendado en el proyecto:
- OULAD: entrenamiento y evaluación del modelo predictivo.
- INE Bolivia: contextualización nacional y análisis de variables educativas,
  laborales y socioeconómicas.
