import torch
import logging
import numpy as np


BATCH_SIZE = 64
EPOCHS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train(model, train_loader, optimizer, criterion):
    train_loss = []

    logger.info(f'Starting training...')
    for epoch in range(EPOCHS):
        model.train()

        epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs = data[0].float().to(DEVICE)
            labels = data[1].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(torch.long))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        train_loss.append(epoch_loss/len(train_loader))
        logger.info(f'\t|| Epoch {epoch} with loss {train_loss[-1]: .5f} ||')
    
    return np.array(train_loss)


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

    