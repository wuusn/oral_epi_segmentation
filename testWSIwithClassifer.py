from unet import UNet
import torch
from glob import glob
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn
import sys
import large_image
import time
import png
import os
from matplotlib import cm
from utils.dataset import *
from torch.utils.data import DataLoader

psize=256
batch_size=4#10
scale = 4
downby = 4
savedir = '/mnt/md0/_datasets/OralCavity/WSI/downby16/SFVA_mini'
model_path = sys.argv[1]
target = sys.argv[2]
input_mode = sys.argv[3]
if input_mode == 'tma':
    psize = 1024
#mark = model_path.split('/')[-3] + '_' + model_path[-6:-4]
mark = 'test_classifer'

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
net = torch.load(model_path, map_location=device)
net.eval()

#classifer_path = '/mnt/md0/_datasets/OralCavity/tma@10_2/patch@256_classifer/save/classfier_2/checkpoints/CP_epoch10.pth'
classifer_path = '/mnt/md0/andrew/PytorchDigitalPathology/classification_lymphoma_densenet/oral_dense_densenet_best_model.pth' 
from torchvision.models import DenseNet
checkpoint = torch.load(classifer_path, map_location=device)
model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint['block_config'], num_init_features=checkpoint['num_init_features'], bn_size=checkpoint['bn_size'], drop_rate=checkpoint['drop_rate'],num_classes=checkpoint['num_classes']).to(device)
classifer = model
n_classes = 2
def testOneWSI(im_path):
    im_name = im_path.split('/')[-1]
    maskname = f'{savedir}/{im_name}'
    maskname = maskname.replace('.tif', f'_pred_{mark}.png')
    #if os.path.exists(maskname):
    #    print(maskname, 'already done')
    #    return
    try:
        ts = large_image.getTileSource(im_path)
        w = ts.sizeX
        h = ts.sizeY
    except Exception as e:
        print(e)
        return
    dataset = WSIDataset(ts, input_mode)
    #print('length:', len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=20, pin_memory=True)
    #Heat = np.zeros((h//scale//downby +psize//downby,w//scale//downby +psize//downby, 3)).astype(np.uint8)
    #Mask = np.zeros((h//scale//downby +psize//downby,w//scale//downby +psize//downby)).astype(np.uint8)
    Heat = np.zeros((h//scale//downby +psize,w//scale//downby +psize, 3)).astype(np.uint8)
    Ori = np.zeros((h//scale//downby +psize,w//scale//downby +psize, 3)).astype(np.uint8)
    Mask = np.zeros((h//scale//downby +psize,w//scale//downby +psize)).astype(np.uint8)

    I = 0
    for batch in loader:
        imgs = batch['image']
        imgs = imgs.to(device=device, dtype=torch.float32)
        output = net(imgs)
        output_class = classifer(imgs)
        output = output.squeeze(1)
        #output_class = output_class.squeeze(1)
        if net.n_classes > 1:
            out  = output.detach().cpu().numpy()
            pred = p
            cpredflat = np.argmax(p, axis=1).flatten()
            #output = np.moveaxis(output,0,-1)
            #mask_pred = np.argmax(output, axis=1)
            pred_class = cpredflat

        else:
            output = torch.sigmoid(output)
            output_class = output_class.detach().cpu().numpy()
            pred = output.detach().cpu().numpy()
            pred_class = np.argmax(output_class, axis=1).flatten()
            mask_pred = (pred>.5).astype(np.uint8) # b,h,w
            
        
        nb = pred.shape[0]
        #count += nb
        for i in range(nb):
            drop = True
            if(pred_class[i]>.5):
                drop = False
            jj = I//math.ceil(h/psize/scale)
            ii = I%math.ceil(h/psize/scale)
            top = jj * psize * scale
            left = ii * psize * scale
            if(drop):
                pred[i] = pred[i]*0
                mask_pred[i] = mask_pred[i] * 0
            jet_pred = cm.jet(pred[i])[:,:,:3]*255
            jet_pred = jet_pred.astype(np.uint8)
            bool_mask = pred[i]>.5
            img = batch['ori'][i]#imgs[i].detach().cpu().numpy()
            #img = np.moveaxis(img, 0, -1)
            heat = np.copy(img)
            heat[bool_mask] = jet_pred[bool_mask]
            heat = heat.astype(np.uint8)
            img = np.copy(img)
            img = img.astype(np.uint8)
            #
            pil_heat = Image.fromarray(heat)
            pil_img = Image.fromarray(img)
            pil_heat = pil_heat.resize((psize//downby, psize//downby), Image.BICUBIC)
            pil_img = pil_img.resize((psize//downby, psize//downby), Image.BICUBIC)
            np_heat = np.array(pil_heat).astype(np.uint8)
            np_img = np.array(pil_img).astype(np.uint8)

            pil_mask_pred = Image.fromarray(mask_pred[i]*255)
            pil_mask_pred = pil_mask_pred.resize((psize//downby, psize//downby), Image.BICUBIC)
            np_mask_pred = (np.array(pil_mask_pred)/255).astype(np.uint8)
            #print(np_mask_pred.shape)
            #
            #print(h,w,left,top)
            Mask[left//scale//downby:left//scale//downby+psize//downby, top//scale//downby:top//scale//downby+psize//downby] = np_mask_pred
            Heat[left//scale//downby:left//scale//downby+psize//downby, top//scale//downby:top//scale//downby+psize//downby,:] = np_heat
            Ori[left//scale//downby:left//scale//downby+psize//downby, top//scale//downby:top//scale//downby+psize//downby,:] = np_img 
            I = I +1
    #
    new_im_name = im_name.replace('.tif', '.png')
    heat_name = im_name.replace('.tif', f'_heat_{mark}.png')
    Image.fromarray(Heat[:h//scale//downby, :w//scale//downby]).save(f'{savedir}/{heat_name}')
    #Image.fromarray(Ori[:h//scale//downby, :w//scale//downby]).save(f'{savedir}/{new_im_name}')
    png.from_array(Mask[:h//scale//downby, :w//scale//downby]*255, mode='L').save(maskname)
    print(maskname, 'done')
    #print('tA:', tA)
    #print('tB:', tB)
    #print('tC:', tC)





def testOneWSI_v1(im_path):
    tA = 0
    tB = 0
    tC = 0
    im_name = im_path.split('/')[-1]
    maskname = f'{savedir}/{im_name}'
    maskname = maskname.replace('.tif', f'_pred_{mark}.png')
    if os.path.exists(maskname):
        print(maskname, 'already done')
        return
    try:
        ts = large_image.getTileSource(im_path)
    except Exception as e:
        print(e)
        return

    w = ts.sizeX
    h = ts.sizeY
    M = np.zeros((h//scale//downby +psize//downby,w//scale//downby +psize//downby)).astype(np.uint8)
    A = np.zeros((h//scale//downby +psize//downby,w//scale//downby +psize//downby, 3)).astype(np.uint8)
    B = np.zeros((h//scale//downby +psize//downby,w//scale//downby +psize//downby, 3)).astype(np.uint8)
    count = 0
    for j in range(0,h,psize*scale):
        for i in range(0,w,psize*scale):
            start = time.time()
            #patch = pil_im.crop((i,j,i+psize,j+psize))
            patch, _ = ts.getRegion(
                    region = dict(left=i, top=j, width=psize*scale, height=psize*scale),
                    format = large_image.tilesource.TILE_FORMAT_PIL
                    )
            end = time.time()
            tA = tA + (end-start)/60
            start = time.time()
            count = count + 1
            patch = patch.convert(mode='RGB')
            patch = patch.resize((psize,psize), Image.BICUBIC)
            np_patch = np.array(patch).astype(np.uint8)
            downby_patch = patch.resize((psize//downby, psize//downby), Image.BICUBIC)
            end = time.time()
            tB = tB + (end-start)/60
            start = time.time()
            np_downby_patch = np.array(downby_patch).astype(np.uint8)
            #tensor_patch = transforms.ToTensor()(np_patch)
            if input_mode == 'gradient':
                np_patch = get_gradient(np_patch)
                np_patch = np_patch.reshape(1,psize,psize)
            else:
                np_patch = np_patch.transpose((2,0,1))
            np_patch = np_patch / 255
            tensor_patch = torch.from_numpy(np_patch)
            x = tensor_patch.unsqueeze(0)
            x = x.to(device, dtype=torch.float32)
            end = time.time()
            tC = tC + (end-start)/60
            #
            output = net(x)
            if net.n_classes > 1:
                output = output.detach().squeeze().cpu().numpy()
                output = np.moveaxis(output,0,-1)
                mask_pred = np.argmax(output, axis=2)
            else:
                output = torch.sigmoid(output)
                pred = output.detach().squeeze().cpu().numpy()
                mask_pred = (pred>.5).astype(np.uint8)

            #
            jet_pred = cm.jet(pred)[:,:,:3]*255
            jet_pred = jet_pred.astype(np.uint8)
            np_patch = np.array(patch).astype(np.uint8)
            np_patch_heat = np.copy(np_patch)
            np_patch_heat[pred>.5] = jet_pred[pred>.5]
            pil_heat = Image.fromarray(np_patch_heat)
            pil_heat_resized = pil_heat.resize((psize//downby, psize//downby), Image.BICUBIC)
            np_heat_resized = np.array(pil_heat_resized).astype(np.uint8)

            pil_mask_pred = Image.fromarray(mask_pred*255)
            pil_mask_pred_resized = pil_mask_pred.resize((psize//downby, psize//downby), Image.BICUBIC)
            np_mask_pred_resized = (np.array(pil_mask_pred_resized)/255).astype(np.uint8)

            M[j//scale//downby:j//scale//downby+psize//downby,i//scale//downby:i//scale//downby+psize//downby] = np_mask_pred_resized
            A[j//scale//downby:j//scale//downby+psize//downby,i//scale//downby:i//scale//downby+psize//downby,:] = np_downby_patch
            B[j//scale//downby:j//scale//downby+psize//downby,i//scale//downby:i//scale//downby+psize//downby,:] = np_heat_resized
    #pil_M = Image.fromarray(M[:h//scale,:w//scale]*255)
    #pil_M.save(maskname)
    new_im_name = im_name.replace('.tif', '.png')
    #print('#####')
    #print(A[:h//scale//downby,:w//scale//downby, :].shape)
    #png.from_array(A[:h//scale//downby,:w//scale//downby, :], mode='RGB').save(f'{savedir}/{new_im_name}')
    Image.fromarray(A[:h//scale//downby,:w//scale//downby]).save(f'{savedir}/{new_im_name}')
    heat_name = im_name.replace('.tif', f'_heat_{mark}.png')
    #png.from_array(B[:h//scale//downby,:w//scale//downby, :], mode='RGB').save(f'{savedir}/{heat_name}')
    #Image.fromarray(B[:h//scale//downby,:w//scale//downby]).save(f'{savedir}/{heat_name}')
    print('tA:', tA)
    print('tB:', tB)
    print('tC:', tC)
    #png.from_array(M[:h//scale//downby,:w//scale//downby]*255, mode='L').save(maskname)
    print(maskname, "done")
    print(count)
if __name__ == '__main__':

    start = time.time()
    if target.endswith('.tif'):
        testOneWSI(target)
    else:
        #im_paths = glob(f'{target}/*/*.tif')
        im_paths = glob(f'{target}/*.tif')
        for im_path in im_paths:
            testOneWSI(im_path)
    end = time.time()
    print('time:', (end-start)/60)
