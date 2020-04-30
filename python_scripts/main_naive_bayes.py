import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import model_creation
import model_evaluation

# Lectura del dataset preprocesado
df = pd.read_csv('../dataset/ticnn_preprocessed.csv')

# Alteración aleatoria del orden de las entradas
df = df.sample(frac=1)

# Convertimos las variables a listas
texts = list(df['text'])
targets = list(df['type'])

# Dividimos en entrenamiento y pruebas
x_train, y_train, x_test, y_test = model_creation.train_test_split(texts, targets)

# Creamos un feature vector para el cálculo de Naive Bayes
count = CountVectorizer(lowercase=False, binary=False)

# Aplicamos fit_transform al conjunto de entrenamiento y transform al de pruebas
count_x_train = count.fit_transform(x_train)
count_x_test = count.transform(x_test)

# Calculamos las probabilidades
naive_bayes = MultinomialNB()
naive_bayes.fit(count_x_train, y_train)

# Realizamos las predicciones en el conjunto de pruebas
predictions = naive_bayes.predict(count_x_test)

# Evaluamos los resultados
print('----------------------- RESULTADOS -----------------------')
print('Accuracy: ', accuracy_score(y_test, predictions))
print('Precision: ', precision_score(y_test, predictions))
print('Recall: ', recall_score(y_test, predictions))
print('F1-Score: ', f1_score(y_test, predictions))

# Matriz de Confusión
model_evaluation.generate_confusion_matrix(y_test, predictions)
