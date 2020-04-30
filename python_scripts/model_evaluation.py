import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_history(history, accuracy=True, precision=True, recall=True, f1=True, loss=True):
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

def generate_confusion_matrix(results, predictions):
    cm = confusion_matrix(results, predictions)
    sns.heatmap(cm, square=True, annot=True,
                cmap= 'RdBu', cbar=False,
                xticklabels= ['Real', 'Fake'],
                yticklabels= ['Real', 'Fake'])

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()