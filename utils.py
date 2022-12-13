import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy_confusion(y_true, y_pred, title):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')

    s = sns.heatmap(conf_matrix, linewidths=.1, annot=True, square=True)
    s.set(xlabel='Predicted', ylabel='GT', title=f'{title}, accuracy = {accuracy}')
    plt.show()

    return accuracy


def plot_loss(losses, legends):
    for loss, legend in zip(losses, legends):
        x = np.arange(loss.shape[0])
        plt.plot(x, loss, label=legend)
    plt.title('Loss curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def prediction(model, dataloader):
    model.eval()

    labels_true = []
    labels_pred = []

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs = data[0].float().to(DEVICE)
            labels = data[1].to(DEVICE)

            outputs = model(inputs)
            pred = outputs.argmax(axis=1)

            for i in labels:
                labels_true.append(i.item())

            for i in pred:
                labels_pred.append(i.item())

    return np.array(labels_true), np.array(labels_pred)

