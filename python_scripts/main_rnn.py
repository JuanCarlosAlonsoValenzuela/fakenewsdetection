import pandas as pd
import numpy as np
from ast import literal_eval
from datetime import datetime
from tensorflow import keras
from keras.preprocessing.text import Tokenizer as tf_tokenizer
from keras.preprocessing.sequence import pad_sequences

import model_creation
import model_training
import model_evaluation

# ---------------------- CONFIGURACIÓN ----------------------
padded_sequence_len = 600
vocabulary_size = 20000
embedding_dimension = 100
lstm_units = 256
n_epochs = 3
dropout_rate = 0.2
learning_rate = 0.001
batch_size = 50
# -----------------------------------------------------------

# Lectura del dataset preprocesado
df = pd.read_csv('../dataset/ticnn_preprocessed.csv')

# Alteración aleatoria del orden de las entradas
df = df.sample(frac=1)

# Convertimos la columna text de string a lista de string
df['text']=df['text'].apply(literal_eval)

# Aplicamos la función convert_to_string
df['text']=df['text'].apply(model_creation.convert_to_string)

# Convertimos las variables a listas
texts = list(df['text'])
targets = list(df['type'])

# Dividimos en conjunto de entrenamiento y pruebas
x_train, y_train, x_test, y_test = model_creation.train_test_split(texts, targets)

# Creamos el vocabulario
tokenizer = tf_tokenizer(num_words=vocabulary_size, oov_token='<null_token>', lower=False, char_level=False)
tokenizer.fit_on_texts(x_train)

# Generamos las secuencias
x_train_sequence = tokenizer.texts_to_sequences(x_train)
x_test_sequence = tokenizer.texts_to_sequences(x_test)

# Padding de las secuencias
train_padded = pad_sequences(x_train_sequence, maxlen=padded_sequence_len,
                             dtype='int32', truncating='post', padding='post')
test_padded = pad_sequences(x_test_sequence, maxlen=padded_sequence_len,
                            dtype='int32', truncating='post', padding='post')

# Creación del modelo neuronal
model = model_training.generate_rnn(voc_size=vocabulary_size,
                                    emb_dim=embedding_dimension,
                                    lstm_u=lstm_units,
                                    lr=learning_rate,
                                    dr=dropout_rate)

# Definimos la ruta en la que se guardarán los logs de TensorBoard
logdir = "logs\\rnn\\" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Definimos el callback de TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# Entrenamos el modelo
history = model.fit(train_padded,
                    np.array(y_train),
                    epochs=n_epochs,
                    validation_data=(test_padded,np.array(y_test)),
                    callbacks=[tensorboard_callback],
                    batch_size=batch_size)

# Representación gráfica de los resultados
model_evaluation.plot_history(history)
