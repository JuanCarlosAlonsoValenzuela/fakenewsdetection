import pandas as pd
import numpy as np
from tensorflow import keras
from datetime import datetime
from ast import literal_eval
from gensim.models import Word2Vec

import model_creation
import model_training
import model_evaluation

# ---------------------- CONFIGURACIÓN ----------------------
vocabulary_size = 20000
# Posibles valores: skipgram, cbow
modelo = 'cbow'
epocas = 8
batch_size = 20
# -----------------------------------------------------------

# Lectura del dataset preprocesado
df = pd.read_csv('../dataset/ticnn_preprocessed.csv')

# Alteracion aleatoria del orden de las entradas
df = df.sample(frac=1)

# Convertimos la columna text a lista de strings
df['text'] = df['text'].apply(literal_eval)

# Convertimos las variables a listas
texts = list(df['text'])
targets = list(df['type'])

# Dividimos en conjunto de entrenamiento y pruebas
x_train, y_train, x_test, y_test = model_creation.train_test_split(texts, targets)

# Creamos el modelo
sg_value = 0
if modelo == 'skipgram':
    sg_value = 1

[print('La variante seleccionada es Skigram') if sg_value == 1 else print('La variante seleccionada es Continuous Bag of Words')]

# Obtenemos los Word Embeddings del modelo
print('Calculando vocabulario Word2Vec (Este paso puede tomar varios minutos)')
model = Word2Vec(x_train, min_count=5, size=300, window=5, iter=10, sg=sg_value)

# Aplicamos word average a todos los documentos de ambos conjuntos
print('Aplicando Word average a los conjuntos de entrenamiento y pruebas...')
x_train_dense = []
for doc in x_train:
    dense_doc = model_training.word_average(doc, model)
    x_train_dense.append(dense_doc)

x_test_dense = []
for doc in x_test:
    dense_doc = model_training.word_average(doc, model)
    x_test_dense.append(dense_doc)

print('Word average finalizado')

# Generamos el modelo neuronal
model = model_training.generate_w2v_dense_nn()
print(model.summary())

# Definimos la ruta en la que se guardarán los logs de TensorBoard
logdir = "logs\\w2v\\" + modelo + "\\" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Definimos el callback de TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# Entrenamos el modelo y guardamos los resultados
history = model.fit(np.array(x_train_dense),
                    np.array(y_train),
                    epochs=epocas,
                    validation_data=(np.array(x_test_dense), np.array(y_test)),
                    callbacks=[tensorboard_callback],
                    batch_size=batch_size)

# Representación gráfica
model_evaluation.plot_history(history)
