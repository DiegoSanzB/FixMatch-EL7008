import torch
import logging
import numpy as np

BATCH_SIZE = 1024
EPOCHS = 100
PATIENCE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train(model, train_loader, val_loader, optimizer, criterion, checkpoint_path):
    train_loss = []
    val_loss = []
    
    min_val_loss = 1e5
    epoch_of_min_loss = 0

    logger.info(f'Starting training...')
    for epoch in range(EPOCHS):
        # Process train loader
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

        # Validation with val_loader
        model.eval()
        loss_epoch_val = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs_val = data[0].float().to(DEVICE)
                labels_val = data[1].to(DEVICE)

                outputs = model(inputs_val)
                loss = criterion(outputs, labels_val.to(torch.long))
                loss_epoch_val += loss.item()

        val_loss.append(loss_epoch_val/len(val_loader))

        # Info of current epoch
        logger.info(f'\t|| Epoch {epoch:02d} with training loss {train_loss[-1]: .5f} and validation loss {val_loss[-1]: .5f} ||')

        # Patience implementation
        if min_val_loss > val_loss[-1]:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss[-1],
                'val_loss': val_loss[-1]
            }, checkpoint_path)
            min_val_loss = val_loss[-1]
            epoch_of_min_loss = epoch

        if epoch > (epoch_of_min_loss + PATIENCE):
            logger.info(f'\t|| Early stop at epoch {epoch}, loading model from best epoch ||')
            break
    
        
    # Load model of best epoch    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_cp = checkpoint['epoch']
    train_loss_cp = checkpoint['train_loss']
    val_loss_cp = checkpoint['val_loss']

    logger.info(f' Loaded model from epoch {epoch_cp} with train loss {train_loss_cp: .5f} and val loss {val_loss_cp: .5f} ')

    return np.array(train_loss), np.array(val_loss)

    