import pandas as pd
import tensorflow as tf

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


def train_test_split_balanced(dataframe):
    # Diferenciar las noticias según el tipo
    df_real = dataframe.loc[dataframe['type'] == 0.0]
    df_fake = dataframe.loc[dataframe['type'] == 1.0]

    # Cambiar orden aleatoriamente
    df_real = df_real.sample(frac=1)
    df_fake = df_fake.sample(frac=1)

    train_test_split = int(len(dataframe) * 0.7)

    # Hacemos una copia para realizar la división en entr. y pruebas
    df_real_copy = df_real.copy()
    df_fake_copy = df_fake.copy()

    # Noticias verdaderas
    df_real_train = df_real_copy.sample(int(train_test_split // 2))
    df_real_test = df_real_copy.drop(df_real_train.index)

    # Noticias falsas
    df_fake_train = df_fake_copy.sample(int(train_test_split // 2))
    df_fake_test = df_fake_copy.drop(df_fake_train.index)

    # Unimos ambos dataframes
    df_train = pd.concat([df_real_train, df_fake_train], axis=0)
    df_test = pd.concat([df_real_test, df_fake_test], axis=0)

    # Alteramos el orden aleatoriamente
    df_train = df_train.sample(frac=1)
    df_test = df_test.sample(frac=1)

    x_train = list(df_train['text'])
    y_train = list(df_train['type'])

    x_test = list(df_test['text'])
    y_test = list(df_test['type'])

    return x_train, y_train, x_test, y_test

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


def convert_to_string(received_input):
    res = ''
    for word in received_input:
        res = res + ' ' + word
    return res
