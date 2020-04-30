import nltk
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

import nltk
nltk.download('stopwords')

pos_map = {
    'CC': 'n', 'CD': 'n', 'DT': 'n', 'EX': 'n', 'FW': 'n', 'IN': 'n', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'LS': 'n',
    'MD': 'v', 'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'PDT': 'n', 'POS': 'n', 'PRP': 'n', 'PRP$': 'r',
    'RB': 'r',
    'RBR': 'r', 'RBS': 'r', 'RP': 'n', 'TO': 'n', 'UH': 'n', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v',
    'VBZ': 'v', 'WDT': 'n', 'WP': 'n', 'WP$': 'n', 'WRB': 'r'
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tokenizer(example_sent):
    # Capitalization
    example_sent = str(example_sent).lower()
    # HTML TAGS
    example_sent = BeautifulSoup(example_sent, 'lxml').text
    # EMAIL ADDRESSES
    example_sent = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', example_sent)
    # URLs
    example_sent = re.sub(r'http\S+', '', example_sent)
    # Punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(example_sent)
    # POS Tagging
    tags = nltk.pos_tag(word_tokens)
    # Lemmatization
    for i, word in enumerate(word_tokens):
        word_tokens[i] = lemmatizer.lemmatize(word, pos=pos_map.get(tags[i][1], 'n'))

    # stop words
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    # digits
    filtered_sentence = [w for w in filtered_sentence if not w.isdigit()]

    return filtered_sentence


def chunk_filtering(df):
    # Mantenemos solamente las columnas que vamos a utilizar
    df = df[['type', 'title', 'content']]

    # Nos quedamos con las noticias fake y reliable
    df = df.loc[(df['type'] == 'fake') | (df['type'] == 'reliable')]

    # Eliminamos las columnas que contienen null values
    df = df.dropna()

    return df


# F1-Score no está entre las métricas por defecto definidas en tf, por lo que definimos una función que nos permita implementarla
from keras import backend as k


# Es una media armónica entre precision y recall
# Como no podemos utilizar tensores a la hora de definir una métrica en modo Graph execution, es necesario volver a calcular precision y recall
# Afortunadamente, es un cálculo sencillo que se realiza rápidamente


def f1_score(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())

    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())

    return 2 * ((precision * recall) / (precision + recall + k.epsilon()))


# Esta función realiza el split entre conjunto de entrenamiento y conjunto de pruebas
def train_test_split(texts, targets):
  # Dividimos en conjunto de entrenamiento y pruebas con una distribución de
  # un 70% de los datos para entrenamiento y un 30% para pruebas
  train_test_split = int(len(targets)*0.7)

  x_train = texts[:train_test_split]
  y_train = targets[:train_test_split]

  x_test = texts[train_test_split:]
  y_test = targets[train_test_split:]

  return x_train, y_train, x_test, y_test



# Esta función nos permite imprimir las representaciones gráficas de las métricas y la función de pérdidas

def plot_history(history, accuracy=True, precision=True, recall=True,
                 f1=True, loss=True):
    plt.style.use('ggplot')

    # Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    x = range(1, len(acc) + 1)

    if accuracy:
        plt.ylim(0., 1.)
        plt.plot(x, acc, 'b', label='Accuracy en entrenamiento')
        plt.plot(x, val_acc, 'r', label='Accuracy en pruebas')
        plt.title('Accuracy en entrenamiento y pruebas')
        plt.legend()

        plt.show()

    # Precision
    if precision:
        precision = history.history['Precision']
        val_precision = history.history['val_Precision']

        plt.ylim(0., 1.)
        plt.plot(x, precision, 'b', label='Precision en entrenamiento')
        plt.plot(x, val_precision, 'r', label='Precision en pruebas')
        plt.title('Precision en entrenamiento y pruebas')
        plt.legend()

        plt.show()

    # Recall
    if recall:
        recall = history.history['Recall']
        val_recall = history.history['val_Recall']

        plt.ylim(0., 1.)
        plt.plot(x, recall, 'b', label='Recall en entrenamiento')
        plt.plot(x, val_recall, 'r', label='Recall en pruebas')
        plt.title('Recall en entrenamiento y pruebas')
        plt.legend()

        plt.show()

    # F1-Score
    if f1:
        f1_score = history.history['f1_score']
        val_f1_score = history.history['val_f1_score']

        plt.ylim(0., 1.)
        plt.plot(x, f1_score, 'b', label='F1-Score en entrenamiento')
        plt.plot(x, val_f1_score, 'r', label='F1-Score en pruebas')
        plt.title('F1-Score en entrenamiento y pruebas')
        plt.legend()

        plt.show()

    # Loss
    if loss:
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(x, loss, 'b', label='Vector de pérdidas en entrenamiento')
        plt.plot(x, val_loss, 'r', label='Vector de pérdidas en pruebas')
        plt.title('Vector de pérdidas en entrenamiento y pruebas')
        plt.legend()

        plt.show()


# Modelo regresión logística
def generate_logistic_regression(dimension):
    # Modelo secuencial
    model = tf.keras.models.Sequential()

    # Capa de entrada
    model.add(tf.keras.layers.InputLayer(input_shape=(dimension,),
                                         sparse=True,
                                         batch_size=20,
                                         dtype=tf.float64))

    # Regresión logística
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', dtype=tf.float64))

    # Compilamos el modelo
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', f1_score])

    return model


# Modelo Word2Vec
def generate_w2v_dense_nn():
  # Modelo secuencial
  model = tf.keras.models.Sequential()

  # Capa de entrada
  model.add(tf.keras.layers.InputLayer(input_shape = (300, ), 
                                     dtype = tf.float32))
  
  # Capa densa
  model.add(
      tf.keras.layers.Dense(
          units = 150,
          activation = 'relu'
      )
  )

  # Capa Dropout
  model.add(
      tf.keras.layers.Dropout(
          rate = 0.2
      )
  )
  
  # Regresión logística
  model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', dtype = tf.float32))

  # Compilamos el modelo
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
                metrics=['accuracy', 'Precision', 'Recall', f1_score])

  return model


# Modelo Doc2Vec
def generate_d2v_dense_nn():
  # Modelo secuencial
  model = tf.keras.models.Sequential()

  # Capa de entrada
  model.add(tf.keras.layers.InputLayer(input_shape = (300, ),  
                                     dtype = tf.float32))
  
  # Capa fully connected
  model.add(
      tf.keras.layers.Dense(
          units = 150,
          activation = 'relu'
      )
  )

  # Capa Dropout
  model.add(
      tf.keras.layers.Dropout(
          rate = 0.2
      )
  )

  # Regresión logística
  model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', dtype = tf.float32))

  # Compilamos el modelo
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
                metrics=['accuracy', 'Precision', 'Recall', f1_score])

  return model



# Convertir a vector sparse
# Obtenemos la representación densa (en forma de matrix)
def convert_to_sparse_tensor(matrix):
  # Convertimos a numpy
  sparse_binary = matrix.todense()
  del matrix

  # Convertimos a tensor
  tensor_binary = tf.convert_to_tensor(sparse_binary, dtype = tf.float64)
  del sparse_binary

  # Convertimos a tensor sparse (menor almacenamiento)
  tensor_sparse = tf.sparse.from_dense(tensor_binary)
  del tensor_binary

  return tensor_sparse



def generate_rnn(voc_size, emb_dim, lstm_u):
  model = tf.keras.Sequential()
  
  # Capa de Embedding
  model.add(
    tf.keras.layers.Embedding(input_dim = voc_size,
                              output_dim = emb_dim,
                              embeddings_initializer = 'uniform',
                              mask_zero = True)
    )

  # Capa bidireccional de LSTM
  model.add(
      # Capa Bidireccional
      tf.keras.layers.Bidirectional(
          # Capa LSTM
          tf.keras.layers.LSTM(units = lstm_u,
                             activation = 'tanh',   
                             recurrent_activation = 'sigmoid',
                             use_bias = True,
                             dropout = 0.2,
                             recurrent_dropout = 0.05
                             )
        )
    )

  # Capa Densa número 1
  model.add(
      tf.keras.layers.Dense(
          units = 128,
          activation = 'relu'
      )
  )

  # Capa Dropout número 1
  model.add(
      tf.keras.layers.Dropout(
          rate = 0.2
      )
  )

  # Capa Densa número 2
  model.add(
      tf.keras.layers.Dense(
          units = 64,
          activation = 'relu'
      )
  )

  # Capa Dropout número 2
  model.add(
      tf.keras.layers.Dropout(
          rate = 0.2
      )
  )

  # Capa de salida
  model.add(
      tf.keras.layers.Dense(
          units = 1,
          activation = 'sigmoid'
      )
  )


  # Compilamos el modelo
  model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
                  metrics=['accuracy', 'Precision', 'Recall', f1_score])
  
  return model
  
  
 # Calcula la media de los word embeddings recibidos por parámetro
def word_average(doc, model):
    mean = []
    for word in doc:
        if word in model.wv.vocab:
            mean.append(model.wv.get_vector(word))
    if not mean:
        # Si el texto está vacío devuelve un vector de ceros
        return np.zeros(model.vector_size)
    else:
        mean = np.array(mean).mean(axis=0)
        return mean

def convert_to_string(received_input):
    res = ''
    for word in received_input:
        res = res + ' ' + word
    return res

def vector_for_learning(model, docs):
    sents = docs
    targets, feature_vectors = zip(*[(doc.tags[0],
                                      model.infer_vector(doc.words, steps=20)) for doc in sents])

    return targets, feature_vectors