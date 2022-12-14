import torch
import torch.nn as nn
import logging
import numpy as np
from data.augmentation import SoftAugment, RandAugment, MyAugment
from utils import DEVICE

EPOCHS = 100
PATIENCE = 10
N, M = 3, 10

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_fixmatch(model, supervised_loader, unsupervised_loader, val_loader, optimizer, 
                    criterion, checkpoint_path, tau=0.95, lambda_u=1):
    logger.info(f'Starting fixmatch training...')
    # Initialize arrays for collecting metrics
    supervised_losses = []
    unsupervised_losses = []
    mask_rate = []
    mask_rate_class = [tau]
    impurity = []
    val_loss = []

    # Define augmentations
    soft_augmentation = SoftAugment()
    strong_augmentation = MyAugment(N, M)
    softmax = nn.Softmax(dim=1)

    # Initialize variables for patience early stop
    min_val_loss = 1e5
    epoch_of_min_loss = 0

    for epoch in range(EPOCHS):
        model.train()
        # Create iterators for loaders (must have same len)
        iter_supervised = iter(supervised_loader)
        iter_unsupervised = iter(unsupervised_loader)   

        epoch_loss_supervised = 0
        epoch_loss_unsupervised = 0

        surpassed_threshold = 0
        incorrect_predictions = 0

        epoch_soft_max = np.empty(0)
        epoch_soft_argmax = np.empty(0)

        for i in range(len(iter_supervised)):
            # Get current batches for supervised (s) and unsupervised (u) data and labels
            inputs_s, labels_s = next(iter_supervised)
            inputs_s = inputs_s.float().to(DEVICE)
            labels_s = labels_s.to(DEVICE)
            
            inputs_u, labels_u = next(iter_unsupervised)
            inputs_u = inputs_u.float()
            labels_u = labels_u.to(DEVICE)  # we don't use the real unsupervised labels during training, only for impurity calculation

            # compute supervised loss   
            optimizer.zero_grad()
            supervised_outputs = model(inputs_s)
            supervised_loss = criterion(supervised_outputs, labels_s.to(torch.long))
            epoch_loss_supervised += supervised_loss.item()

            # compute unsupervised loss
            unsupervised_loss = 0

            # Soft and strong augment current unsupervised batch
            soft_tensor = torch.empty(0).to(DEVICE)
            strong_tensor = torch.empty(0).to(DEVICE)
            for j in range(inputs_u.shape[0]):
                img = inputs_u[j, :, :, :]
                soft_img = torch.tensor(soft_augmentation(img)).float().to(DEVICE)
                strong_img = torch.tensor(strong_augmentation(img)).float().to(DEVICE)

                soft_tensor = torch.cat((soft_tensor, torch.unsqueeze(soft_img, dim=0)), dim=0)
                strong_tensor = torch.cat((strong_tensor, torch.unsqueeze(strong_img, dim=0)), dim=0)
            
            soft_tensor = soft_tensor.float().to(DEVICE)
            strong_tensor = strong_tensor.float().to(DEVICE)
            
            # Process soft augmented data
            soft_output = model(soft_tensor)
            soft_output = softmax(soft_output)
            
            # Create pseudo labels
            soft_max, soft_argmax = torch.max(soft_output, dim=1)
            soft_threshold = soft_max.ge(tau).float().to(DEVICE)
            
            # Compute impurity and mask rate
            surpassed_threshold += soft_threshold.sum()
            incorrect_predictions += ((soft_argmax.ne(labels_u)) * soft_threshold).sum()
            
            # Process strong augmented data
            strong_output = model(strong_tensor)
            
            # Get loss from strong augmentation given the pseudo labels
            unsupervised_loss = criterion(strong_output, soft_argmax) * soft_threshold
            unsupervised_loss = unsupervised_loss.mean()     
            
            epoch_loss_unsupervised += unsupervised_loss.item()
            
            # backwards with weighted sum of losses
            loss = supervised_loss + unsupervised_loss * lambda_u
            loss.backward()
            optimizer.step()

            epoch_soft_max = np.hstack((epoch_soft_max, soft_max.cpu().detach().numpy()))
            epoch_soft_argmax = np.hstack((epoch_soft_argmax, soft_argmax.cpu().detach().numpy()))

        # Append all data necessary for metrics calculation        
        supervised_losses.append(epoch_loss_supervised/len(iter_supervised))
        unsupervised_losses.append(epoch_loss_unsupervised/len(iter_supervised))

        impurity.append((incorrect_predictions / surpassed_threshold).item())
        mask_rate.append((surpassed_threshold / inputs_u.shape[0]).item() / len(iter_unsupervised))

        mask_rate_class.append((epoch_soft_max, epoch_soft_argmax))

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

        # Patience implementation
        if min_val_loss > val_loss[-1]:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'supervised_loss': supervised_losses[-1],
                'unsupervised_loss': unsupervised_losses[-1],
                'val_loss': val_loss[-1]
            }, checkpoint_path)
            min_val_loss = val_loss[-1]
            epoch_of_min_loss = epoch

        logger.info(f'\t|| Epoch {epoch:02d} with supervised loss {supervised_losses[-1]: .5f}, unsupervised loss {unsupervised_losses[-1]: .5f} and validation loss {val_loss[-1]: .5f} ||')

        if epoch > (epoch_of_min_loss + PATIENCE):
            logger.info(f'\t|| Early stop at epoch {epoch}, loading model from best epoch ||')
            break

    # Load model of best epoch
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_cp = checkpoint['epoch']
    
    logger.info(f' Loaded model from epoch {epoch_cp}')

    return np.array(supervised_losses), np.array(unsupervised_losses), np.array(val_loss), np.array(impurity), np.array(mask_rate), mask_rate_class




