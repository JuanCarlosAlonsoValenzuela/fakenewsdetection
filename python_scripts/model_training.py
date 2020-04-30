import tensorflow as tf
import os
import numpy as np

import nlp_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
                  metrics=['accuracy', 'Precision', 'Recall', nlp_utils.f1_score])

    return model


# Modelo Word2Vec
def generate_w2v_dense_nn():
    # Modelo secuencial
    model = tf.keras.models.Sequential()

    # Capa de entrada
    model.add(tf.keras.layers.InputLayer(input_shape=(300,),
                                         dtype=tf.float32))

    # Capa densa
    model.add(
        tf.keras.layers.Dense(
            units=150,
            activation='relu'
        )
    )

    # Capa Dropout
    model.add(
        tf.keras.layers.Dropout(
            rate=0.2
        )
    )

    # Regresión logística
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', dtype=tf.float32))

    # Compilamos el modelo
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', nlp_utils.f1_score])

    return model


# Modelo Doc2Vec
def generate_d2v_dense_nn():
    # Modelo secuencial
    model = tf.keras.models.Sequential()

    # Capa de entrada
    model.add(tf.keras.layers.InputLayer(input_shape=(300,),
                                         dtype=tf.float32))

    # Capa fully connected
    model.add(
        tf.keras.layers.Dense(
            units=150,
            activation='relu'
        )
    )

    # Capa Dropout
    model.add(
        tf.keras.layers.Dropout(
            rate=0.2
        )
    )

    # Regresión logística
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', dtype=tf.float32))

    # Compilamos el modelo
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', nlp_utils.f1_score])

    return model


# Generar Red Neuronal Recurrente
def generate_rnn(voc_size, emb_dim, lstm_u, lr, dr):
    model = tf.keras.Sequential()

    # Capa de Embedding
    model.add(
        tf.keras.layers.Embedding(input_dim=voc_size,
                                  output_dim=emb_dim,
                                  embeddings_initializer='uniform',
                                  mask_zero=True)
    )

    # Capa bidireccional de LSTM
    model.add(
        # Capa Bidireccional
        tf.keras.layers.Bidirectional(
            # Capa LSTM
            tf.keras.layers.LSTM(units=lstm_u,
                                 activation='tanh',
                                 recurrent_activation='sigmoid',
                                 use_bias=True,
                                 dropout=dr,
                                 recurrent_dropout=0.05
                                 )
        )
    )

    # Capa Densa número 1
    model.add(
        tf.keras.layers.Dense(
            units=128,
            activation='relu'
        )
    )

    # Capa Dropout número 1
    model.add(
        tf.keras.layers.Dropout(
            rate=dr
        )
    )

    # Capa Densa número 2
    model.add(
        tf.keras.layers.Dense(
            units=64,
            activation='relu'
        )
    )

    # Capa Dropout número 2
    model.add(
        tf.keras.layers.Dropout(
            rate=dr
        )
    )

    # Capa de salida
    model.add(
        tf.keras.layers.Dense(
            units=1,
            activation='sigmoid'
        )
    )

    # Definimos el optimizador
    rmsprop_optim = tf.keras.optimizers.RMSprop(learning_rate=lr)

    # Compilamos el modelo
    model.compile(optimizer=rmsprop_optim, loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', nlp_utils.f1_score])

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



def vector_for_learning(model, docs):
    sents = docs
    targets, feature_vectors = zip(*[(doc.tags[0],
                                      model.infer_vector(doc.words, steps=20)) for doc in sents])

    return targets, feature_vectors