import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import nlp_utils


def run(run_dir, hparams, train_padded, test_padded, y_train, y_test):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)   # Almacenar los valores de los parámetros usados
        f1 = train_test_rnn(hparams, train_padded, test_padded, y_train, y_test)  # Entrenamos el modelo con los valores especificados
        tf.summary.scalar('f1-score', f1, step=1)


def train_test_rnn(hparams, train_padded, test_padded, y_train, y_test):
    # ----Comenzamos definiendo el modelo
    model = tf.keras.Sequential()
    # Capa de Embedding
    model.add(tf.keras.layers.Embedding(input_dim=20000,  # Tamaño del vocabulario
                                        output_dim=100,  # Número de dimensiones de WE
                                        embeddings_initializer='uniform',
                                        mask_zero=True))

    # Capa bidireccional
    model.add(tf.keras.layers.Bidirectional(
        # Capa LSTM
        tf.keras.layers.LSTM(units=128,  # Hiperparámetro a optimizar
                             activation='tanh',
                             recurrent_activation='sigmoid',
                             use_bias=True,
                             dropout=hparams[next(iter(hparams))],  # Hiperparámetro a optimizar
                             recurrent_dropout=0.05)))
    # Capa densa 1
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Capa Dropout 1
    model.add(tf.keras.layers.Dropout(rate=hparams[next(iter(hparams))]))  # Hiperparámetro a optimizar

    # Capa densa 2
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))

    # Capa Dropout 2
    model.add(tf.keras.layers.Dropout(rate=hparams[next(iter(hparams))]))  # Hiperparámetro a optimizar

    # Capa de salida
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Definimos el optimizador
    rmsprop_optim = tf.keras.optimizers.RMSprop(learning_rate=0.001)  # Hiperparámetro a optimizar

    # ----Compilamos el modelo
    model.compile(optimizer=rmsprop_optim, loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', nlp_utils.f1_score])

    # ----Entrenar el modelo (No es necesario llamar a tensorboard)
    model.fit(train_padded, np.array(y_train), epochs=1,
              batch_size=200)  # Hiperparámetro a optimizar

    # ----Evaluar el conjunto de pruebas
    # loss, acc, pre, rec, f1
    _, _, _, _, f1 = model.evaluate(test_padded, np.array(y_test))

    return f1