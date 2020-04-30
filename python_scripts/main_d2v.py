import pandas as pd
import numpy as np
from ast import literal_eval
import multiprocessing
from tensorflow import keras
from datetime import datetime
from sklearn import utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import model_creation
import model_training
import model_evaluation

# ---------------------- CONFIGURACIÓN ----------------------
# Posibles valores: dm, dbow
modelo = 'dbow'
epocas = 7
batch_size = 20
n_dimensions = 300
# -----------------------------------------------------------

# Lectura del dataset preprocesado
df = pd.read_csv('../dataset/ticnn_preprocessed.csv')

dm_value = 0
if modelo == 'dm':
    dm_value = 1

[print('La variante seleccionada es Distributed Memory') if dm_value == 1 else print('La variante seleccionada es Distributed Bag of Words')]


# Alteracion aleatoria del orden de las entradas
df = df.sample(frac=1)

# Convertimos la columna text a lista de strings
df['text'] = df['text'].apply(literal_eval)

# Convertimos las variables a listas
texts = list(df['text'])
targets = list(df['type'])

# Dividimos en conjunto de entrenamiento y pruebas
x_train, y_train, x_test, y_test = model_creation.train_test_split(texts, targets)

# Generamos los TaggedDocuments
train_documents = []
test_documents = []

for i in range(len(x_train)):
    train_documents.append(
        TaggedDocument(words=x_train[i], tags=[y_train[i]])
    )

for i in range(len(x_test)):
    test_documents.append(
        TaggedDocument(words= x_test[i], tags=[y_test[i]])
    )

# Creamos el modelo Doc2Vec
print('Creando el modelo Doc2Vec')
d2v_model = Doc2Vec(dm=dm_value,
                    vector_size=n_dimensions,
                    min_count=2,
                    workers=multiprocessing.cpu_count())

# Construimos el vocabulario utilizando solamente el conjunto de entrenamiento
print('Construyendo el vocabulario...')
d2v_model.build_vocab([x for x in train_documents])

# Entrenamos el modelo Doc2Vec
print('Entrenamos el modelo Doc2Vec')
train_documents = utils.shuffle(train_documents)
d2v_model.train(train_documents,
                total_examples=len(train_documents),
                epochs=10)

# Aplicamos la función vector for learning
print('Aplicando la función vector for learning... (Este paso tomará varios minutos)')
y_train, x_train = model_training.vector_for_learning(d2v_model, train_documents)
y_test, x_test = model_training.vector_for_learning(d2v_model, test_documents)

# Generamos el modelo neuronal
model = model_training.generate_d2v_dense_nn()
print(model.summary())

# Definimos la ruta en la que se guardarán los logs de TensorBoard
logdir = "logs\\d2v\\" + modelo + "\\" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Definimos el callback de TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# Entrenamos el modelo
history = model.fit(np.array(x_train),
                    np.array(y_train),
                    epochs=epocas,
                    validation_data=(np.array(x_test), np.array(y_test)),
                    callbacks=[tensorboard_callback]
                    )

# Representación gráfica de los resultados
model_evaluation.plot_history(history)
