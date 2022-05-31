import torch
from unet import UNet
import os
from glob import glob
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import torch.nn as nn
import sys
from matplotlib import cm


def mask2rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            p = mask[i, j]
            p = int(p)
            color = label2color[p]
            rgb[i, j] = color
    rgb = rgb.astype(np.uint8)
    return rgb


def testOneTMA(im_path):
    name = im_path.split('/')[-1].replace(src_ext, '')
    pil_im = Image.open(im_path)

    w, h = pil_im.size
    w = w // scale
    h = h // scale
    pil_im = pil_im.resize((w, h), Image.BICUBIC)
    M = np.zeros((h + psize, w + psize)).astype(np.uint8)
    Heat = np.zeros((h + psize, w + psize, 3)).astype(np.uint8)

    count = 0
    for j in range(0, h, psize):
        for i in range(0, w, psize):
            count += 1
            patch = pil_im.crop((i, j, i + psize, j + psize))

            np_patch = np.array(patch).astype(np.uint8)

            np_patch = np_patch.transpose((2, 0, 1))

            np_patch = np_patch / 255

            tensor_patch = torch.from_numpy(np_patch)

            x = tensor_patch.unsqueeze(0)
            x = x.to(device, dtype=torch.float32)
            output = net(x)
            masks_pred = output

            if net.n_classes > 1:
                output = output.detach().squeeze().cpu().numpy()
                output = np.moveaxis(output, 0, -1)
                mask_pred = np.argmax(output, axis=2)

            else:
                output = torch.sigmoid(output)
                pred = output.detach().squeeze().cpu().numpy()
                mask_pred = (pred > .5).astype(np.uint8)
                bool_mask = pred > .5
                #yflat = y.cpu().numpy().flatten()
                cpredflat = bool_mask.astype(np.uint8).flatten()
                #cmatrix = cmatrix + confusion_matrix(yflat, cpredflat, range(2))
                jet_pred = cm.jet(pred)[:, :, :3] * 255
                jet_pred = jet_pred.astype(np.uint8)
                rgm_im = np.array(patch).astype(np.uint8)  #.convert('RGB')
                rgm_im = np.array(rgm_im)  #/255
                heat = rgm_im
                #print(heat.shape, jet_pred.shape, bool_mask.shape)
                heat[bool_mask] = jet_pred[bool_mask]
                heat = heat.astype(np.uint8)

            M[j:j + psize,
              i:i + psize] = (pred * 255).astype(np.uint8)  #mask_pred
            Heat[j:j + psize, i:i + psize, :] = heat

    pil_M = Image.fromarray(M[:h, :w])
    pil_H = Image.fromarray(Heat[:h, :w, :])
    maskname = f'{save_dir}/{name}_pred.png'
    heatname = f'{save_dir}/{name}_pred_heat.png'
    pil_M.save(maskname)
    pil_H.save(heatname)


if __name__ == '__main__':
    psize = 256
    model_path = sys.argv[1]
    target = sys.argv[2]
    src_ext = sys.argv[3]
    scale = int(sys.argv[4])
    save_dir = sys.argv[5]

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(model_path, map_location=device)
    net.eval()
    label2color = {0: [0, 0, 0], 1: [227, 207, 87], 2: [178, 34, 34]}
    criterion = nn.BCEWithLogitsLoss()

    if target.endswith(src_ext):
        print(testOneTMA(target))
    else:
        im_paths = glob(f'{target}/*{src_ext}')

        for im_path in im_paths:
            testOneTMA(im_path)
