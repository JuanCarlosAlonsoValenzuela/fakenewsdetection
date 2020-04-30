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