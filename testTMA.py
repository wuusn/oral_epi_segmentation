from unet import UNet
import os
import torch
from glob import glob
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn
import sys
import time
from matplotlib import cm

#from post_processing import *
Image.MAX_IMAGE_PIXELS = None

laid=8
psize=256-2*laid
threshold=.5#.65
model_path = sys.argv[1]
target = sys.argv[2]
src_ext = sys.argv[3]
save_dir = sys.argv[4]
os.makedirs(save_dir, exist_ok=True)
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
net = torch.load(model_path, map_location=device)
net.eval()
label2color = {0:[0,0,0], 1:[227,207,87], 2:[178, 34, 34]}
criterion = nn.BCEWithLogitsLoss()
def mask2rgb(mask):
    h,w = mask.shape
    rgb = np.zeros((h,w,3))
    for i in range(h):
        for j in range(w):
            p = mask[i,j]
            p = int(p)
            color = label2color[p]
            rgb[i,j] = color
    rgb = rgb.astype(np.uint8)
    return rgb

def testOneTMA(im_path):
    name = im_path.split('/')[-1].replace(f'.{src_ext}', '')
    pil_im = Image.open(im_path)
    #pil_im.save(f'{save_dir}/{name}.png')
    w,h = pil_im.size
    M = np.zeros((h+psize,w+psize)).astype(np.uint8)
    Heat = np.zeros((h+psize,w+psize, 3)).astype(np.uint8)

    for j in range(0,h,psize):
        for i in range(0,w,psize):
            patch = pil_im.crop((i-laid,j-laid,i+psize+laid,j+psize+laid))
            np_patch = np.array(patch).astype(np.uint8)
            np_patch = np_patch.transpose((2,0,1))
            np_patch = np_patch / 255
            tensor_patch = torch.from_numpy(np_patch)
            x = tensor_patch.unsqueeze(0)
            x = x.to(device, dtype=torch.float32)
            output = net(x)
            output = output[:,:,laid:laid+psize,laid:laid+psize]
            masks_pred = output
            if net.n_classes > 1:
                output = output.detach().squeeze().cpu().numpy()
                output = np.moveaxis(output,0,-1)
                mask_pred = np.argmax(output, axis=2)
            else:
                output = torch.sigmoid(output)
                pred = output.detach().squeeze().cpu().numpy()
                mask_pred = (pred>threshold).astype(np.uint8)
                bool_mask = pred>threshold
                jet_pred = cm.jet(pred)[:,:,:3]*255
                jet_pred = jet_pred.astype(np.uint8)
                rgm_im = np.array(patch).astype(np.uint8)#.convert('RGB')
                rgm_im = np.array(rgm_im)#/255
                heat = rgm_im
                heat = heat[laid:laid+psize, laid:laid+psize]
                #print(heat.shape, jet_pred.shape, bool_mask.shape)
                heat[bool_mask] = jet_pred[bool_mask]
                heat = heat.astype(np.uint8)

            M[j:j+psize,i:i+psize] = mask_pred
            Heat[j:j+psize,i:i+psize,:] = heat
    #M = post_processing(M)
    pil_M = Image.fromarray(M[:h,:w]*255)
    pil_H = Image.fromarray(Heat[:h,:w,:])
    maskname = f'{save_dir}/{name}_pred.png'
    heatname = f'{save_dir}/{name}_pred_heat.png'
    pil_M.save(maskname)
    pil_H.save(heatname)

if __name__ == '__main__':
    start=time.time()
    if target.endswith(src_ext):
        testOneTMA(target)
    else:
        im_paths = glob(f'{target}/*{src_ext}')
        for im_path in im_paths:
            print(im_path.split('/')[-1])
            testOneTMA(im_path)
    end=time.time()
    print('done:', (end-start)/60)

