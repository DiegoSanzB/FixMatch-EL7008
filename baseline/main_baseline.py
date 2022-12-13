import sys
sys.path.append('..')

import logging
from data.dataset import get_cifar10, CIFAR10
from models.vgg import VGG16, VGG9
from models.myNet import MyNet, NetCN4
from models.vgg_cifar import VGG11_BN
from utils import accuracy_confusion, plot_loss, prediction
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from train_baseline import BATCH_SIZE, EPOCHS, DEVICE
from train_baseline import train

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
        train_size=0.95, random_state=SEED, shuffle=SHUFFLE
    )
    supervised_data, unsupervised_data, supervised_labels, unsupervised_labels = train_test_split(
        train_data, train_labels, 
        train_size=0.1, random_state=SEED, shuffle=SHUFFLE
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
    net = VGG11_BN()
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    train_loss, val_loss = train(net, train_loader, val_loader, optimizer, criterion, CHECKPOINT_PATH)

    labels_true, labels_pred = prediction(net, test_loader)
    
    logger.info(f'Generating confusion matrix')
    accuracy = accuracy_confusion(labels_true, labels_pred, 'Baseline confusion matrix', 'baseline_cm.png')
    logger.info(f'Accuracy: {accuracy: .5f}')

    logger.info(f'Plotting loss curves')
    plot_loss((train_loss, val_loss), ('Train', 'Val'), 'baseline_loss.png')

