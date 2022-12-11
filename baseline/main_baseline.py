import sys
sys.path.append('..')

import logging
from data.dataset import get_cifar10, CIFAR10
from models.vgg import VGG16, VGG9
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train_baseline import BATCH_SIZE, EPOCHS, DEVICE
from train_baseline import train, prediction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logger.info(f'Using Device: {DEVICE}')
    # setup dataset and dataloader
    train_data, train_labels, test_data, test_labels = get_cifar10()
    
    Cifar10_train = CIFAR10(train_data, train_labels)
    Cifar10_test = CIFAR10(test_data, test_labels)
    train_loader = DataLoader(Cifar10_train, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(Cifar10_test, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True)
    
    # initialize model vgg16, criterion and optimizer
    Vgg9 = VGG9()
    Vgg9.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Vgg9.parameters(), lr=1e-3)

    train_loss = train(Vgg9, train_loader, optimizer, criterion)

    labels_true, labels_pred = prediction(Vgg9, test_loader)
    
    confusion_matrix_norm = confusion_matrix(labels_true, labels_pred, normalize='true')

    s = sns.heatmap(confusion_matrix_norm, linewidths=.1, annot=True, square=True)
    s.set(xlabel='Predicho', ylabel='Real', title = f'Matriz de confusion')
    plt.show()