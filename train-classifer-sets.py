import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net, eval_net_class
from unet import *

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, TMADataset, ClassDataset
from torch.utils.data import DataLoader, random_split


#base_dir= '/mnt/md0/_datasets/OralCavity/tma@10_2'
base_dir = '/mnt/md0/_datasets/OralCavity/tma@10_2/patch@256_classifer_2/sets' 
#base_dir= '/mnt/md0/_datasets/OralCavity/tma_3'
n_classes = 1

def train_net(net,
              device,
              epochs=50,
              batch_size=14,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              args=False):

    edge = args.edge
    mark = args.mark
    cmark = args.cmark
    start_epoch = args.start_epoch
    #dir_save = f'{base_dir}/{args.data}/save/{mark}'
    dir_save = f'/mnt/md0/_datasets/OralCavity/tma@10_2/patch@256_classifer_2/save/{mark}' 
    dir_checkpoint = f'{dir_save}/checkpoints'
    os.makedirs(dir_checkpoint, exist_ok=True)
    #train = BasicDataset(f'{base_dir}/{args.data}/train', 'train', args)
    #val = BasicDataset(f'{base_dir}/{args.data}/val', 'val', args)
    #train = ClassDataset(f'{base_dir}/train', 'train', args)
    #val = ClassDataset(f'{base_dir}/val', 'val', args)
    #n_train = len(train)
    #n_val = len(val)
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=18, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size//2, shuffle=False, num_workers=18, pin_memory=True)


    date_str = time.strftime('%m%d-%H:%M')
    writer = SummaryWriter(f'{dir_save}/runs/{date_str}')
    edge_weight = 1.1

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

        #Training size:   {n_train}
        #Validation size: {n_val}

    #if n_classes > 1:
    #    optimizer = optim.Adam(net.parameters())
    #else:
    #    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    #    #optimizer = optim.RMSprop(net.parameters(), lr=lr)

    optimizer = optim.Adam(net.parameters())

    #weights = torch.tensor([1.,10.,1.]).to(device)
    if n_classes > 1:
        #criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    loss_history={'train':[], 'val':[]}
    global_step = 0
    if edge:
        edge_weight = torch.tensor(edge_weight).to(device)

    I=0
    for epoch in range(start_epoch, epochs+start_epoch):
        train = ClassDataset(f'{base_dir}/set-{I}/train', 'train', args)
        val = ClassDataset(f'{base_dir}/set-{I}/val', 'val', args)
        I = I+1
        if I==26:
            I = 0
        n_train = len(train)
        n_val = len(val)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=18, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size//2, shuffle=False, num_workers=18, pin_memory=True)
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                labels = batch['label']
                masks_weight = batch['edge_weight']
                assert imgs.shape[1] == n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                #mask_type = torch.float32 if net.n_classes == 1 else torch.long
                #true_masks = true_masks.to(device=device, dtype=mask_type)
                labels = labels.to(device, dtype=torch.float32)

                output = net(imgs)
                #if net.n_classes==1:
                #    masks_pred = masks_pred.squeeze(1)
                output = output.squeeze(1)

                loss = criterion(output, labels)
                if edge:
                    masks_weight = masks_weight.to(device=device, dtype=torch.float32)
                    loss = (loss * (edge_weight**masks_weight)).mean()
                epoch_loss += loss.item()*imgs.shape[0]

                writer.add_scalar('Loss/train', loss.item(), global_step)
                global_step +=1
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])

        epoch_loss = epoch_loss / n_train
        val_loss, cmatrix = eval_net_class(net, val_loader, device, n_val)
        acc = (cmatrix/cmatrix.sum()).trace()
        loss_history['train'].append(epoch_loss)
        loss_history['val'].append(val_loss)

        writer.add_scalar(f'val/acc', acc, epoch)
        writer.add_scalar(f'val/TN', cmatrix[0,0], epoch)
        writer.add_scalar(f'val/TP', cmatrix[1,1], epoch)
        writer.add_scalar(f'val/FP', cmatrix[0,1], epoch)
        writer.add_scalar(f'val/FN', cmatrix[1,0], epoch)
        writer.add_scalar(f'val/TNR', cmatrix[0,0]/(cmatrix[0,0]+cmatrix[0,1]), epoch)
        writer.add_scalar(f'val/TPR', cmatrix[1,1]/(cmatrix[1,1]+cmatrix[1,0]), epoch)

        if n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_loss))
            writer.add_scalar('Loss/val', val_loss, epoch)

        else:
            logging.info('Validation Loss: {}'.format(val_loss))
            writer.add_scalar('Loss/val', val_loss, epoch)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.to(torch.device('cpu')), dir_checkpoint + f'/CP_epoch{epoch + 1}.pth')
            pickle.dump({'loss':loss_history}, open(f'{dir_save}/hisotry.p', 'wb'))

            net.to(device)
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(1, N+1), loss_history["train"], label="train_loss")
    plt.plot(np.arange(1, N+1), loss_history["val"], label="val_loss")
    plt.title("Training Loss and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.savefig(f'{dir_save}/plot.png')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-n', '--n-classes', metavar='N', type=int, nargs='?', default=1,
                        help='n classes', dest='nclasses')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-c', '--cmark', dest='cmark', type=str, default=False,
                        help='cmark filter')
    parser.add_argument('-z', '--frozen', dest='frozen', type=str, default=False,
                        help='frozen layers for finetune')
    parser.add_argument('-m', '--mark', dest='mark', type=str, default=False, help='mark')
    parser.add_argument('--edge', dest='edge', type=str, default=False,
                        help='loss with edge weight')
    parser.add_argument('--colorau', dest='colorau', type=str, default=True,
                        help='color augmentation')
    parser.add_argument('--he_color', dest='he_color', type=str, default=False,
                        help='he color augmentation')
    parser.add_argument('--gradient', dest='gradient', type=str, default=False,
                        help='covert rgb to r gradient img')
    parser.add_argument('--data', dest='data', type=str, default='patch@256',
                        help='dataset dir')
    parser.add_argument('--nnmodel', dest='nnmodel', type=str, default='unet',
                        help='nn model name')
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=0,
                        help='start epoch')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.cmark == 'True':
        args.cmark = True
    if args.frozen == 'True':
        args.frozen = True
    if args.colorau == 'True':
        args.colorau = True
    if args.gradient == 'True':
        args.gradient = True
    if args.he_color == 'True':
        args.he_color = True
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    if args.gradient:
        n_channel = 1
    else:
        n_channel = 3
    n_channels = n_channel
    if args.nnmodel == 'yus':
        net = Yus_Unet(n_channel, args.nclasses).to(device)
    elif args.nnmodel == 'unet':
        net = UNet(n_channels=n_channel, n_classes=args.nclasses, bilinear=False, bn=True).to(device)
    elif args.nnmodel == 'vgg':
        net = get_vgg()
        net = net.to(device)
    else:
        net = TestNet(n_channels=n_channel, n_classes=args.nclasses).to(device)
    if(args.frozen == True):
        for param in net.parameters():
            param.requires_grad = False
        #for param in net.up3.parameters():
        #    param.requires_grad = True
        for param in net.up4.parameters():
            param.requires_grad = True
        for param in net.outc.parameters():
            param.requires_grad = True
    logging.info(f'Network:\n'
                 f'\t{n_channel} input channels\n'
                 f'\t{args.nclasses} output channels (classes)\n')
                 #f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net = torch.load(args.load, map_location=device)
        logging.info(f'Model loaded from {args.load}')

    if args.edge:
        logging.info(f'edge weighted on loss')
        args.edge = True
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  args =args)
    except KeyboardInterrupt:
        os.makedirs(dir_checkpoint, exist_ok=True)
        torch.save(net.to(torch.device('cpu')).state_dict(), f'{dir_checkpoint}/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
