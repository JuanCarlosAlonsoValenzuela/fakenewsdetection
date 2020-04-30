import pandas as pd
import preprocessing
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer as tf_tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorboard.plugins.hparams import api as hp
import scipy

import model_creation
import model_training
import hparams_optimization

import sys
sys.path.insert(0, '../dataset/')

#####################################################################################################
# PRUEBAS SPRINT 1
#####################################################################################################
def prueba1():
    print("--- Prueba 1: Lectura del fichero csv. \n")
    print("El archivo subset_test.csv debe encontarse en la misma carpeta que este notebook")

    try:
        datos = pd.read_csv('subset_test.csv')
    except:
        print('Error al leer el fichero csv')
        raise

    print("El fichero se ha leído correctamente. Los primeros cinco elementos son:")
    print(datos.head())
    print('-' * 40)
    print('Los últimos 5 elementos son:')
    print(datos.tail())
    print('\n-------------------- Prueba 1 finalizada con éxito --------------------\n')


def prueba2():
    print('--- Prueba 2: Preprocesado del dataset \n')
    try:
        print("Lectura del fichero... \n")
        datos = pd.read_csv('subset_test.csv')
    except:
        print("Error al leer el fichero csv")
        raise
    print("El fichero se ha leído correctamente")

    print("Primeras tres entradas sin preprocesar")
    print(datos.head(3))
    print('-' * 40)

    try:
        print("Preprocesado del dataset (Puede tardar algunos minutos)")
        datos['content'] = datos['content'].apply(preprocessing.tokenizer)
    except:
        print("Error al preprocesar el conjunto de datos")
        raise
    print("El preprocesado se ha realizado correctamente")
    print("Primeras tres entradas preprocesadas")
    print(datos.head(3))
    print('\n-------------------- Prueba 2 finalizada con éxito --------------------\n')


def prueba3():
    test = 123456.6

    try:
        print('--- Prueba 3: Preprocesado de un objeto que no es de tipo String')
        test = preprocessing.tokenizer(test)
    except:
        print("Error al preprocesar una entrada que no es de tipo String")
        raise

    print('El preprocesado se ha realizado correctamente')
    print(test)
    print('\n-------------------- Prueba 3 finalizada con éxito --------------------\n')


def prueba4():
    test = "<HTML>This <p>is.a</p> ! jua@email.com sentences, showing off the <br> stop words filtration. http://www.youtube.com 68509"

    try:
        print('--- Prueba 4: Preprocesado de un string')
        test = preprocessing.tokenizer(test)
    except:
        print('Error al preprocesar el string')
        raise
    print('El preprocesado se ha realizado correctamente')
    print(test)
    print('\n-------------------- Prueba 4 finalizada con éxito --------------------\n')


#####################################################################################################
# PRUEBAS SPRINT 2
#####################################################################################################
def prueba5():
    print('Prueba 5: División del dataset en conjunto de entrenamiento y pruebas')

    try:
        print('\n--- Lectura del dataset preprocesado')
        df = pd.read_csv('../dataset/ticnn_preprocessed.csv')
    except:
        print('Se ha producido un error al leer el dataset')
        raise

    print('Dataset leído correctamente, mostrando las tres primeras entradas')
    print(df.head(3))

    try:
        print('\n--- División de los datos en entrenamiento y pruebas')
        print('70% entrenamiento - 30% pruebas')

        texts = list(df['text'])
        targets = list(df['type'])

        x_train, y_train, x_test, y_test = model_creation.train_test_split(texts, targets)
        print('Datos separados correctamente')

    except:
        print('Error al separar en entrenamiento y pruebas')
        raise

    try:
        print('\n--- Comprobamos que el tamaño de los conjuntos es correcto')

        assert len(x_train) == int(len(df) * 0.7)
        assert len(x_train) == len(y_train)
        print('El tamaño del conjunto de entrenamiento es el correcto')

        assert len(x_test) == int(math.ceil(len(df) * 0.3))
        assert len(x_test) == len(y_test)
        print('El tamaño del conjunto de pruebas es el correcto')

    except:
        print('El tamaño de los conjuntos no es el adecuado')
        raise

    print('\n-------------------------------------------------------------------')
    print('División en conjunto de entrenamiento y pruebas realizada correctamente')
    print('\n-------------------- Prueba 5 finalizada con éxito --------------------\n')


def prueba6():
    print('Prueba 6: Convertir un array de numpy a un sparse vector')

    try:
        print('\n--- Conversión a sparse tensor')

        test_vector = np.zeros(shape=(50000,))
        test_vector_size = sys.getsizeof(test_vector)
        print('El tamaño de la matriz de numpy es {}'.format(test_vector_size))

        test_vector = scipy.sparse.csc_matrix(test_vector)

        sparse_vector = model_creation.convert_to_sparse_tensor(test_vector)
        sparse_vector_size = sys.getsizeof(sparse_vector)

        print('El tamaño de la matriz sparse es {}'.format(sparse_vector_size))

        print('La matriz sparse es de tipo {}'.format(type(sparse_vector)))

        assert isinstance(sparse_vector,
                          type(tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))

        assert test_vector_size > sparse_vector_size
        print('El tamaño del vector sparse es menor que el de la matriz de numpy')

    except:
        print('\nSe ha producido un error al intentar crear el vector sparse')
        raise

    print('\n-------------------------------------------------------------------')
    print('Tranformación a Matriz Sparse realizada correctamente')
    print('\n-------------------- Prueba 6 finalizada con éxito --------------------\n')


def prueba7():
    print('Prueba 7: Creación de un Modelo de Regresión logística')

    try:
        print('\nCreación del modelo...')
        model = model_training.generate_logistic_regression(20000)

    except:
        print('Error al crear el modelo')
        raise

    print('\n------------------------------------')
    print('Modelo creado:')
    print(model.summary())
    print('\nClase del modelo: {}'.format(type(model)))
    print('\n-------------------- Prueba 7 finalizada con éxito --------------------\n')


def prueba8():
    print('Prueba 8: Creación del Modelo Neuronal de Word2Vec')

    try:
        print('\nCreación del Modelo...')
        model = model_training.generate_w2v_dense_nn()

    except:
        print('Error al crear el modelo')
        raise

    print('\n------------------------------------')
    print('Modelo creado:')
    print(model.summary())
    print('\nClase del modelo: {}'.format(type(model)))
    print('\n-------------------- Prueba 8 finalizada con éxito --------------------\n')


def prueba9():
    print('Prueba 9: Creación del Modelo Neuronal de Doc2Vec')

    try:
        print('\nCreación del Modelo...')
        model = model_training.generate_d2v_dense_nn()

    except:
        print('Error al crear el modelo')
        raise

    print('\n------------------------------------')
    print('Modelo creado:')
    print(model.summary())
    print('\nClase del modelo: {}'.format(type(model)))
    print('\n-------------------- Prueba 9 finalizada con éxito --------------------\n')


#####################################################################################################
# PRUEBAS SPRINT 3
#####################################################################################################
def prueba10():
    print('Prueba 10: Creación de la Red Neuronal Recurrente')

    try:
        print('\nCreación del Modelo...')
        model = model_training.generate_rnn(20000, 100, 128)

    except:
        print('Error al crear el modelo')
        raise

    print('\n--------------------------------------')
    print('Modelo creado con éxito')
    print(model.summary())
    print('\nClase del modelo: {}'.format(type(model)))
    print('\n-------------------- Prueba 10 finalizada con éxito --------------------\n')


#####################################################################################################
# PRUEBAS SPRINT 4
#####################################################################################################
def prueba11():
    print('Prueba 11: Grid Search con un subset de los hiperparámetros')

    try:
        print('\n--- Lectura del dataset preprocesado')
        df = pd.read_csv('../dataset/ticnn_preprocessed.csv')
    except:
        print('Se ha producido un error al leer el dataset')
        raise

    print('Dataset Leído correctamente')

    try:
        print('\n--- División de los datos en entrenamiento y pruebas')
        # Alteración del orden
        df = df.sample(frac=1.)
        # Transformamos la columna text a lista de string
        from ast import literal_eval
        df['text'] = df['text'].apply(literal_eval)

        # Aplicamos la función convert_to_string a los textos
        df['text'] = df['text'].apply(model_creation.convert_to_string)

        # Convertimos los textos y las targets variables a listas
        texts = list(df['text'])
        targets = list(df['type'])

        # División en conjunto de entrenamiento y test
        x_train, y_train, x_test, y_test = model_creation.train_test_split(texts, targets)

    except:
        print('Error al separar en entrenamiento y pruebas')
        raise

    print('Datos separados correctamente')

    try:
        print('\n--- Creación del diccionario (Vocabulario)')
        tokenizer = tf_tokenizer(num_words=20000, oov_token='<null_token>',
                                 lower=False, char_level=False)
        tokenizer.fit_on_texts(x_train)

    except:
        print('Error al crear el vocabulario')
        raise

    print('Vocabulario creado correctamente')

    try:
        print('\n--- Transformar los textos en secuencias y padding')
        x_train_sequence = tokenizer.texts_to_sequences(x_train)
        x_test_sequence = tokenizer.texts_to_sequences(x_test)

        train_padded = pad_sequences(x_train_sequence, maxlen=600,
                                     dtype='int32', truncating='post',
                                     padding='post')
        test_padded = pad_sequences(x_test_sequence, maxlen=600,
                                    dtype='int32', truncating='post',
                                    padding='post')

    except:
        print('Error al convertir los textos en secuencias')
        raise

    print('Secuenciación y padding realizados con éxito')

    try:
        print('\n--- Definición de los hiperparámetros a optimizar')
        HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.4))
        METRIC_F1 = 'f1-score'

        with tf.summary.create_file_writer('logs/hparam_tuning(test11)').as_default():
            hp.hparams_config(
                hparams=[HP_DROPOUT],
                metrics=[hp.Metric(METRIC_F1, display_name='F1-Score')])
    except:
        print('Error al definir los hiperparámetros')
        raise
    print('Hiperparámetros definidos correctamente')

    try:
        print('\n--- Ejecución Grid Search')
        session_num = 0

        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            hparams = {HP_DROPOUT: dropout_rate}

            run_name = 'run-%d' % session_num
            print('--Iniciando ejecución : %s' % run_name)
            print({h: hparams[h] for h in hparams})
            hparams_optimization.run('logs/hparam_tuning(test11)/' + run_name,
                hparams, train_padded, test_padded, y_train, y_test)
            session_num += 1
    except:
        print('Error al realizar Grid Search')
        raise

    print('\n-------------------------------------------------------------------')
    print('Grid Search realizado con éxito')