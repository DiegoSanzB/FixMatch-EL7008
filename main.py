import logging
from data.dataset import get_cifar10, CIFAR10
from models.vgg import VGG16, VGG9
from models.vgg_cifar import VGG11_BN
from models.myNet import MyNet, NetCN4
from utils import accuracy_confusion, plot_loss, plot_impurity_mask_rate, prediction
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from train import train_fixmatch
from train import EPOCHS, PATIENCE
from utils import DEVICE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SEED = 5
SHUFFLE = False
CHECKPOINT_PATH = 'best_fixmatch_checkpoint.pt'
BATCH_SIZE = 1024 # For validation and test loaders
LOADER_LENGHT = 50  # Same for supervised and unsupervised loaders

if __name__ == '__main__':
    logger.info(f'Using Device: {DEVICE}')
    # obtain data
    train_data, train_labels, test_data, test_labels = get_cifar10()
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels,
        train_size=0.95, random_state=SEED, shuffle=SHUFFLE
    )
    supervised_data, unsupervised_data, supervised_labels, unsupervised_labels = train_test_split(
        train_data, train_labels,
        train_size=0.15, random_state=SEED, shuffle=SHUFFLE
    )
    # setup datasets
    Cifar10_supervised = CIFAR10(supervised_data, supervised_labels)
    Cifar10_unsupervised = CIFAR10(unsupervised_data, unsupervised_labels)
    Cifar10_val = CIFAR10(val_data, val_labels)
    Cifar10_test = CIFAR10(test_data, test_labels)

    logger.info(f'Size of supervised dataset {len(Cifar10_supervised)}, unsupervised {len(Cifar10_unsupervised)} and validation {len(Cifar10_val)}')

    supervised_batch_size = int(np.ceil(len(Cifar10_supervised) / LOADER_LENGHT))
    unsupervised_batch_size = int(np.ceil(len(Cifar10_unsupervised) / LOADER_LENGHT))

    logger.info(f' with supervised batches of size {supervised_batch_size} and unsupervised {unsupervised_batch_size}')
    # setup dataloaders
    supervised_loader = DataLoader(Cifar10_supervised, batch_size=supervised_batch_size, 
                            shuffle=SHUFFLE, num_workers=2, pin_memory=True)
    unsupervised_loader = DataLoader(Cifar10_unsupervised, batch_size=unsupervised_batch_size, 
                            shuffle=SHUFFLE, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Cifar10_val, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(Cifar10_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                            num_workers=2, pin_memory=True)

    assert len(supervised_loader) == len(unsupervised_loader)

    # initialize model, criterion and optimizer
    net = NetCN4()
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # training
    supervised_loss, unsupervised_loss, val_loss, impurity, mask_rate, mask_rate_class = train_fixmatch(
        net, supervised_loader, unsupervised_loader, val_loader, optimizer,
        criterion, CHECKPOINT_PATH
    )
    train_metrics = np.vstack((supervised_loss, unsupervised_loss, val_loss, impurity, mask_rate))
    
    with open('data.npy', 'wb') as f:
        np.save(f, train_metrics)

    # obtain metrics
    labels_true, labels_pred = prediction(net, test_loader)
    accuracy = accuracy_confusion(labels_true, labels_pred, 'Fixmatch Confusion Matrix', 'fixmatch_cm.png')

    plot_loss((supervised_loss, unsupervised_loss, val_loss), ('Supervised', 'Unsupervised', 'Validation'), 'fixmatch_all_loss.png')

    plot_loss((supervised_loss + unsupervised_loss, val_loss), ('Train', 'Validation'), 'fixmatch_loss.png')

    plot_impurity_mask_rate(impurity, mask_rate, mask_rate_class, 'fixmatch_impurity_mask_rate.png')