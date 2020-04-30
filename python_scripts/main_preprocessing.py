'''

    En este archivo se lleva a cabo el preprocesado del dataset, para que se ejecute correctamente deben cumplirse las siguientes condiciones:
        - El archivo all_data.csv debe encontrarse en la carpeta dataset del proyecto

    En la parte final del script, el dataset preprocesado se exporta a formato .csv, obteniendo el fichero ticnn_preprocessed,
        la línea que ejecuta este comando ha sido comentada para evitar duplicados de dicho fichero, pero si se elimina el comentario
        generará el archivo correctamente.

    En este script se omiten las representaciones gráficas realizadas durante la fase de análisis del conjunto de datos, dichas representaciones
        pueden encontrarse en los notebooks Analisis TI-CNN dataset (Parte 1) y Analisis TI-CNN dataset (Parte 2)
'''

import preprocessing
import pandas as pd

# Cargamos el dataset en memoria
df = pd.read_csv('../dataset/all_data.csv', encoding='utf-8')

# Filtramos el dataset
df = preprocessing.filter_dataset(df)

# Finalmente, preprocesamos el conjunto de datos y lo exportamos
df['text'] = df['text'].apply(preprocessing.tokenizer)
df.to_csv('../dataset/ticnn_preprocessed.csv', index=False)