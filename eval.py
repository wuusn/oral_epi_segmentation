import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

#from dice_loss import dice_coeff
from sklearn.metrics import confusion_matrix
import numpy as np


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    cmatrix = np.zeros((2,2))

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
        cmatrix = np.zeros((net.n_classes,net.n_classes))
    else:
        criterion = nn.BCEWithLogitsLoss()
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            masks_pred = net(imgs)
            if net.n_classes > 1:
                label_range = net.n_classes
                p=masks_pred[:,:,:,:].detach().cpu().numpy()
                cpredflat = np.argmax(p,axis=1).flatten()
                yflat = true_masks.cpu().numpy().flatten()
            else:
                label_range = 2
                masks_pred= masks_pred.squeeze(1)
                p = torch.sigmoid(masks_pred)
                p = p.detach().squeeze().cpu().numpy()
                cpredflat = (p>.5).astype(np.uint8).flatten()
                yflat = true_masks.cpu().numpy().flatten()

            cmatrix = cmatrix + confusion_matrix(yflat, cpredflat, labels=range(label_range))

            loss = criterion(masks_pred, true_masks)
            tot += loss.item()*imgs.shape[0]
            pbar.update(imgs.shape[0])

    return tot / n_val, cmatrix

def eval_net_class(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    cmatrix = np.zeros((2,2))
    n_classes = 2

    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
        cmatrix = np.zeros((n_classes,n_classes))
    else:
        criterion = nn.BCEWithLogitsLoss()
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        label_type = torch.float32 if n_classes == 1 else torch.long
        for batch in loader:
            imgs = batch['image']
            labels = batch['label']

            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=label_type)

            out = net(imgs)
            if n_classes > 1:
                label_range = n_classes
                p=out.detach().cpu().numpy()
                cpredflat = np.argmax(p,axis=1).flatten()
                yflat = labels.cpu().numpy().flatten()
            else:
                label_range = 2
                #masks_pred= masks_pred.squeeze(1)
                p = torch.sigmoid(out)
                p = p.detach().squeeze().cpu().numpy()
                cpredflat = (p>.5).astype(np.uint8).flatten()
                yflat = labels.cpu().numpy().flatten()

            cmatrix = cmatrix + confusion_matrix(yflat, cpredflat, labels=range(label_range))

            #out = out.squeeze(1)
            loss = criterion(out, labels)
            tot += loss.item()*imgs.shape[0]
            pbar.update(imgs.shape[0])

    return tot / n_val, cmatrix

