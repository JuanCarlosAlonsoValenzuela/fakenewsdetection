import tensorflow as tf
import pandas as pd
from datetime import datetime
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import model_creation
import model_training
import model_evaluation


# ---------------------- CONFIGURACIÓN ----------------------
vocabulary_size = 20000
# Posibles valores: binary_count, tf, tfidf
modelo = 'tfidf'
# -----------------------------------------------------------

# Lectura del dataset preprocesado
df = pd.read_csv('../dataset/ticnn_preprocessed.csv')

# Alteración aleatoria del orden de las entradas
df = df.sample(frac=1)

# Convertimos las variables a listas
texts = list(df['text'])
targets = list(df['type'])

# División en conjunto de entrenamiento y pruebas
x_train, y_train, x_test, y_test = model_creation.train_test_split(texts, targets)

# Declaramos el feature_vector, en función de la variante del modelo a aplicar
if modelo == 'binary_count':
    print('La variante seleccionada es Binary Counts')
    feature_vector = CountVectorizer(binary = True, max_features=vocabulary_size, lowercase=False)
elif modelo == 'tf':
    print('La variante seleccionada es Term Frequency')
    feature_vector = TfidfVectorizer(max_features=vocabulary_size, lowercase=False, use_idf=False)
else:   # Asumimos tfidf por defecto
    print('La variante seleccionada es Term Frequency - Inverse Document Frequency')
    feature_vector = TfidfVectorizer(max_features=vocabulary_size, lowercase=False, use_idf=True)


# Aplicamos fit_transform al conjunto de entrenamiento y transform al de pruebas
sparse_texts_train = feature_vector.fit_transform(x_train)
sparse_texts_test = feature_vector.transform(x_test)

# Obtenemos los tensores_sparse para reducir el gasto de memoria
tensor_sparse_train = model_creation.convert_to_sparse_tensor(sparse_texts_train)
tensor_sparse_test = model_creation.convert_to_sparse_tensor(sparse_texts_test)

# Generamos el modelo indicando el tamaño de vocabulario como parámetro
model = model_training.generate_logistic_regression(vocabulary_size)
print(model.summary())

# Definimos la ruta en la que se guardarán los logs de TensorBoard
logdir = "logs\\bow\\" + modelo + "\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Definimos el callback de TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# Entrenamos el modelo y guardamos los resultados en la variable history para realizar la representación gráfica
history = model.fit(tensor_sparse_train, tf.convert_to_tensor(y_train, dtype=tf.float64),
                    epochs = 10,
                    validation_data = (tensor_sparse_test,
                                       tf.convert_to_tensor(y_test, dtype=tf.float64)
                                       ),
                    callbacks = [tensorboard_callback]
                    )

# Representación gráfica
model_evaluation.plot_history(history)
