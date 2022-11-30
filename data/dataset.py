import os
import logging
import numpy as np
import requests
import tarfile
import pickle

# Constants
LINKS = {
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def to_image(image, side):
    return np.reshape(image, (3, side, side))

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_cifar10():
    tar_path = 'cifar-10-python.tar.gz'
    # Download dataset if necessary
    if os.path.exists(tar_path):
        logger.info(f'Tar file allready downloaded')
    else:
        logger.info(f'Retrieving cifar10 dataset')
        with open(tar_path, "wb") as f:
            f.write(requests.get(LINKS['cifar10']).content)

        logger.info(f'Download ready!')

    if os.path.exists('./cifar-10-batches-py'):
        logger.info(f'Tar file allready extracted')
    else:
        logger.info(f'Extracting dataset')
        tar = tarfile.open(tar_path)
        tar.extractall('')
        tar.close()
    # Concatenate all 5 training batches
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_i = unpickle(f'./cifar-10-batches-py/data_batch_{i}')
        train_data.append(batch_i[b'data'])
        train_labels.append(batch_i[b'labels'])
    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)
    # Read test batch
    test_batch = unpickle('./cifar-10-batches-py/test_batch')
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']
    
    return train_data, train_labels, test_data, test_labels

if __name__ == '__main__':
    get_cifar10()

