from unet import UNet
import torch
from glob import glob
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import torch.nn as nn
import sys
from matplotlib import cm

psize=256
model_path = sys.argv[1]
target = sys.argv[2]
mode = sys.argv[3]
mark = model_path.split('/')[-3] + '_' + model_path.split('/')[-1].replace('.pth', '').replace('CP_epoch', '')
ext = '.png'

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
    pil_im = Image.open(im_path)
    mask_path = im_path.replace(ext, '_mask.png')
    pil_mask = Image.open(mask_path)
    phase = im_path.split('/')[-3]
    w,h = pil_im.size
    M = np.zeros((h+psize,w+psize)).astype(np.uint8)
    Heat = np.zeros((h+psize,w+psize, 3)).astype(np.uint8)

    Loss = 0
    cmatrix = np.zeros((2,2))
    count =0
    for j in range(0,h,psize):
        for i in range(0,w,psize):
            count +=1
            patch = pil_im.crop((i,j,i+psize,j+psize))
            patch_mask = pil_mask.crop((i,j,i+psize,j+psize))
            np_patch = np.array(patch).astype(np.uint8)
            if phase == 'nontumor': # epi include nontumor epi
                np_mask= np.array(patch_mask).astype(np.uint8)
                #np_mask = np.zeros((psize,psize))
            else:
                np_mask= np.array(patch_mask).astype(np.uint8)
            np_patch = np_patch.transpose((2,0,1))
            np_patch = np_patch / 255
            np_mask = np_mask / 255 if np.max(np_mask) > 1 else np_mask
            tensor_patch = torch.from_numpy(np_patch)
            y = torch.from_numpy(np_mask)
            y = y.unsqueeze(0)
            y = y.to(device, torch.float32)
            x = tensor_patch.unsqueeze(0)
            x = x.to(device, dtype=torch.float32)
            output = net(x)
            masks_pred = output
            loss = criterion(masks_pred.squeeze(0), y)
            Loss += loss.item()
            if net.n_classes > 1:
                output = output.detach().squeeze().cpu().numpy()
                output = np.moveaxis(output,0,-1)
                mask_pred = np.argmax(output, axis=2)
            else:
                output = torch.sigmoid(output)
                pred = output.detach().squeeze().cpu().numpy()
                mask_pred = (pred>.5).astype(np.uint8)
                bool_mask = pred>.5
                yflat = y.cpu().numpy().flatten()
                cpredflat = bool_mask.astype(np.uint8).flatten()
                cmatrix = cmatrix + confusion_matrix(yflat, cpredflat, range(2))
#                jet_pred = cm.jet(pred)[:,:,:3]*255
#                jet_pred = jet_pred.astype(np.uint8)
#                rgm_im = np.array(patch).astype(np.uint8)#patch.convert('RGB')
#                rgm_im = np.array(rgm_im)#/255
#                heat = rgm_im
                #print(heat.shape, jet_pred.shape, bool_mask.shape)
#                heat[bool_mask] = jet_pred[bool_mask]
#                heat = heat.astype(np.uint8)


#            if mode == 'tma':
#                M[j:j+psize,i:i+psize, :] = mask2rgb(mask_pred)
#            else:
#                M[j:j+psize,i:i+psize] = mask_pred
#                Heat[j:j+psize,i:i+psize,:] = heat
#
#    if mode == 'tma':
#        pil_M = Image.fromarray(M[:h,:w])
#    else:
#        pil_M = Image.fromarray(M[:h,:w]*255)
#        pil_H = Image.fromarray(Heat[:h,:w,:])
#    maskname = im_path.replace(ext, f'_pred_{mark}.png')
#    heatname = im_path.replace(ext, f'_pred_{mark}_heat.png')
    #pil_M.save(maskname)
    #pil_H.save(heatname)
    return Loss/count, cmatrix
if __name__ == '__main__':

    if target.endswith(ext):
        print(testOneTMA(target))
    else:
        cmatrix = np.zeros((2,2))
        im_paths = [f for f in glob(f'{target}/*/*/*{ext}') if not f.endswith('_mask.png')]
        count = 0
        Loss=0
        C= np.zeros((2,2))
        for im_path in im_paths:
            loss, cmatrix = testOneTMA(im_path)
            Loss += loss
            C += cmatrix
            count +=1
        Loss = Loss/ count
        PA = (C/C.sum()).trace()*100
        R = C[1,1]/(C[1,1]+C[1,0])*100
        PPV = C[1,1]/(C[1,1]+C[0,1])*100
        D = 2*C[1,1]/(2*C[1,1]+C[0,1]+C[1,0])*100
        print(mark)
        print('Loss: %.3f  PA: %.3f  R: %.3f  PPV: %.3f  D: %.3f' %(Loss, PA,R,PPV, D))
