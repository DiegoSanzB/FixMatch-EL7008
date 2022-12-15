import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Computes accuracy and confusion matrix for the given prediction and true labels
def accuracy_confusion(y_true, y_pred, title, path):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')

    s = sns.heatmap(conf_matrix, linewidths=.1, annot=True, square=True)
    s.set(xlabel='Predicted', ylabel='GT', title=f'{title}, accuracy = {accuracy}')
    
    plt.savefig(path)
    plt.show()

    return accuracy


# Plots all losses given in a list in one plot
def plot_loss(losses, legends, path):
    for loss, legend in zip(losses, legends):
        x = np.arange(loss.shape[0])
        plt.plot(x, loss, label=legend)
    plt.title('Loss curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.show()


# Plots impurity, mask rate and mask rate by class given each array
def plot_impurity_mask_rate(impurity, mask_rate, mask_rate_class, path):
    f, axis = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(impurity.shape[0])

    # Plot impurity
    axis[0].plot(x, impurity)
    axis[0].set_title('Impurity')

    # Plot Mask rate
    axis[1].plot(x, mask_rate)
    axis[1].set_title('Mask Rate')

    for ax in axis.flat:
        ax.set(xlabel='Epoch')

    # Compute mask rate for each class
    tau = mask_rate_class[0]
    epochs = len(mask_rate_class) - 1
    mask_rate_per_class = np.zeros((10, epochs))
    for i, tup in enumerate(mask_rate_class[1: ]):
        soft_max, soft_argmax = tup
        for j, val in enumerate(soft_max):
            if val > tau:
                mask_rate_per_class[int(soft_argmax[j]), i] += 1
    # Plot mask rate for every class
    for i, row in enumerate(mask_rate_per_class):
        axis[2].plot(x, row, label=f'Class {i}')
    axis[2].set_title('Mask Rate per class')
    axis[2].legend()

    plt.savefig(path)
    plt.show()
    
# Makes the model prediction for the data in the given dataloader
# Returns the true and predicted labels
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

