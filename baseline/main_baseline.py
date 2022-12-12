import sys
sys.path.append('..')

import logging
from data.dataset import get_cifar10, CIFAR10
from models.vgg import VGG16, VGG9
from models.myNet import MyNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from train_baseline import BATCH_SIZE, EPOCHS, DEVICE
from train_baseline import train, prediction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SEED = 5
SHUFFLE = False
CHECKPOINT_PATH = 'best_checkpoint.pt'

if __name__ == '__main__':
    logger.info(f'Using Device: {DEVICE}')
    # setup dataset and dataloader
    train_data, train_labels, test_data, test_labels = get_cifar10()
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, 
        train_size=0.8, random_state=SEED, shuffle=SHUFFLE
    )
    supervised_data, unsupervised_data, supervised_labels, unsupervised_labels = train_test_split(
        train_data, train_labels, 
        train_size=0.2, random_state=SEED, shuffle=SHUFFLE
    )
    

    Cifar10_supervised = CIFAR10(supervised_data, supervised_labels)
    Cifar10_val = CIFAR10(val_data, val_labels)
    Cifar10_test = CIFAR10(test_data, test_labels)

    train_loader = DataLoader(Cifar10_supervised, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(Cifar10_val, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(Cifar10_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                            num_workers=2, pin_memory=True)
    
    # initialize model vgg16/vgg9/myNet, criterion and optimizer
    net = MyNet()
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    train_loss = train(net, train_loader, val_loader, optimizer, criterion, CHECKPOINT_PATH)

    labels_true, labels_pred = prediction(net, test_loader)
    
    logger.info(f'Accuracy: {accuracy_score(labels_true, labels_pred): .5f}')
    logger.info(f'Generating confusion matrix')
    confusion_matrix_norm = confusion_matrix(labels_true, labels_pred, normalize='true')

    s = sns.heatmap(confusion_matrix_norm, linewidths=.1, annot=True, square=True)
    s.set(xlabel='Predicted', ylabel='GT', title = f'Confusion Matrix')
    plt.show()

    plt.plot(np.arange(train_loss.shape[0]), train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.show()