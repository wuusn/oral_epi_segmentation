import torch
import os
from glob import glob
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import torch.nn as nn
import sys
from matplotlib import cm
import multiprocessing
from multiprocessing import Pool, Manager
import signal
from itertools import repeat
import time
import png
import argparse

import large_image
from unet import *
import cv2

Image.MAX_IMAGE_PIXELS = None


def update_state(path, state, tasks):
    tasks[path] = state


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


def handler(signum, frame):
    raise Exception("timeout..")


def testOneWithAlarm(im_path, device, net, args, update, tasks):
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(15 * 60)  # 15 min
        testOne(im_path, device, net, args, update, tasks)
    except Exception as e:
        name = im_path.split('/')[-1].split('.')[:-1]
        print("TimeOut:", name)
    finally:
        signal.alarm(0)


def testOne(im_path, device, net, args, update, tasks):
    mag = args.mag
    psize = args.psize
    laid = args.laid
    threshold = args.threshold
    scale = args.scale
    resize = psize // scale
    model_path = args.model
    save_dir = args.save_dir
    src_ext = args.src_ext
    code = f'_{args.mark}' if args.mark != '' else ''

    name = im_path.split('/')[-1].replace(src_ext, '')

    oripath = f'{save_dir}/{name}.png'
    maskpath = f'{save_dir}/{name}_pred{code}.png'
    heatpath = f'{save_dir}/{name}_pred_heat{code}.png'
    contourpath = f'{save_dir}/{name}{code}.json'

    done = True
    if args.save_heat:
        done = done & os.path.exists(heatpath)
    if args.save_mask:
        done = done & os.path.exists(maskpath)
    if args.save_contour:
        done = done & os.path.exists(contourpath)
    if done:
        #print(name, 'already done!')
        pass
        #return

    print(name)
    isKFB = True if src_ext == '.kfb' else False
    isTIF = True if src_ext == '.tif' or src_ext == '.svs' else False
    if isKFB:
        wsi = QiluWSI(im_path)
        w, h = wsi.size(mag)
    elif isTIF:
        wsi = large_image.getTileSource(im_path)
        mag_scale = args.src_mag // args.mag
        w = wsi.sizeX // mag_scale
        h = wsi.sizeY // mag_scale
    else:
        pil_im = Image.open(im_path)
        w, h = pil_im.size
    Ori = np.zeros(
        (h // scale + psize, w // scale + psize, 3)).astype(np.uint8)
    M = np.zeros((h // scale + psize, w // scale + psize)).astype(np.uint8)
    Heat = np.zeros(
        (h // scale + psize, w // scale + psize, 3)).astype(np.uint8)
    all_count = h // psize * (w // psize)
    count = 0
    #print('for:', h//psize*w//psize)
    for j in range(0, h, psize):
        for i in range(0, w, psize):
            #_start = time.time()
            if isKFB:
                patch = wsi.read_roi_local(i - laid, j - laid,
                                           psize + 2 * laid, psize + 2 * laid)
            elif isTIF:
                patch, _ = wsi.getRegion(
                    region=dict(left=(i - laid) * mag_scale,
                                top=(j - laid) * mag_scale,
                                width=(psize + 2 * laid) * mag_scale,
                                height=(psize + 2 * laid) * mag_scale),
                    format=large_image.tilesource.TILE_FORMAT_PIL)
                #patch,_ = wsi.getRegion(
                #        region = dict(left=i-laid, top=j-laid, width=(psize+2*laid), height=(psize+2*laid)),
                #        format = large_image.tilesource.TILE_FORMAT_PIL)
                #print('read tif patch time:',(time.time()-_start))
                patch = patch.resize((psize + 2 * laid, psize + 2 * laid),
                                     Image.BICUBIC)
                #print('resize patch time:',(time.time()-_start))
            else:
                patch = pil_im.crop(
                    (i - laid, j - laid, i + psize + laid, j + psize + laid))
            np_patch = np.array(patch).astype(np.uint8)
            #print(np_patch.shape)
            assert np_patch.shape == (psize + 2 * laid, psize + 2 * laid, 3)
            # print('read patch time:',(time.time()-_start))
            if isKFB:
                np_patch = np_patch[..., ::-1]  ### bgr 2 rgb
            np_patch = np_patch.transpose((2, 0, 1))
            np_patch = np_patch / 255
            tensor_patch = torch.from_numpy(np_patch)
            x = tensor_patch.unsqueeze(0)
            x = x.to(device, dtype=torch.float32)
            output = net(x)
            output = output[:, :, laid:laid + psize, laid:laid + psize]
            masks_pred = output
            output = torch.sigmoid(output)
            pred = output.detach().squeeze().cpu().numpy()
            mask_pred = (pred > threshold).astype(np.uint8)
            bool_mask = pred > threshold
            cpredflat = bool_mask.astype(np.uint8).flatten()
            jet_pred = cm.jet(pred)[:, :, :3] * 255
            jet_pred = jet_pred.astype(np.uint8)
            rgm_im = np.array(patch).astype(np.uint8)  #.convert('RGB')
            rgm_im = np.array(rgm_im)  #/255
            heat = rgm_im.copy()
            heat = heat[laid:laid + psize, laid:laid + psize]
            heat[bool_mask] = jet_pred[bool_mask]
            heat = heat.astype(np.uint8)
            ori = rgm_im[laid:laid + psize, laid:laid + psize].copy()
            ori_resize = cv2.resize(ori, (resize, resize),
                                    interpolation=cv2.INTER_CUBIC)
            #mask_pred = post_processing(mask_pred)
            mask_pred_resize = cv2.resize(mask_pred, (resize, resize),
                                          interpolation=cv2.INTER_NEAREST)
            heat_resize = cv2.resize(heat, (resize, resize),
                                     interpolation=cv2.INTER_CUBIC)
            Ori[j // scale:j // scale + resize,
                i // scale:i // scale + resize, :] = ori_resize
            M[j // scale:j // scale + resize,
              i // scale:i // scale + resize] = mask_pred_resize
            Heat[j // scale:j // scale + resize,
                 i // scale:i // scale + resize, :] = heat_resize
            count += 1
            state = '{:.0f}%'.format((count) / all_count * 100)
            update(im_path, state, tasks)
            #print('single for time:',(time.time()-_start))

    update(im_path, '生成中', tasks)
    if args.post:
        M = post_processing(M, scale)

    if args.save_ori:
        try:
            #png.from_array(Ori[:h//scale,:w//scale,:], mode='RGB').save(oripath)
            pil_O = Image.fromarray(Ori[:h // scale, :w // scale, :])
            pil_O.save(oripath)
        except:
            try:
                pil_O = Image.fromarray(Ori[:h // scale, :w // scale, :])
                pil_O.save(oripath)
            except Exception as err:
                print(err)
    if args.save_heat:
        try:
            pil_H = Image.fromarray(Heat[:h // scale, :w // scale, :])
            pil_H.save(heatpath)
            #png.from_array(Heat[:h//scale,:w//scale,:], mode='RGB').save(heatpath)
        except:
            try:
                #png.from_array(Heat[:h//scale,:w//scale,:], mode='RGB').save(heatpath)
                pil_H = Image.fromarray(Heat[:h // scale, :w // scale, :])
                pil_H.save(heatpath)
                #pil_H = Image.fromarray(Heat[:h//scale,:w//scale,:])
                #pil_H.save(heatpath)
            except Exception as err:
                print(err)
    if args.save_mask:
        try:
            #png.from_array(M[:h//scale,:w//scale]*255, mode='L').save(maskpath)
            pil_M = Image.fromarray(M[:h // scale, :w // scale] * 255)
            pil_M.save(maskpath)
        except:
            try:
                pil_M = Image.fromarray(M[:h // scale, :w // scale] * 255)
                pil_M.save(maskpath)
            except Exception as err:
                print(err)
    if args.save_contour:
        feature = mask2geojson(M[:h // scale, :w // scale] * 255,
                               upscale=40 // args.mag * scale)
        with open(contourpath, 'w') as f:
            geojson.dump(feature, f, indent=4)
    update(im_path, '已完成', tasks)


def get_args(default=False):
    parser = argparse.ArgumentParser(
        description="generate results based on trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',
                        type=str,
                        help='path of the model file .pth')
    parser.add_argument(
        '--src',
        type=str,
        help='one file or one folder which the model will depoly on')
    parser.add_argument('--src-ext',
                        type=str,
                        help='src filename suffix/extension, like .png, .kfb',
                        default='.kfb')
    parser.add_argument('--save-dir',
                        type=str,
                        help='the directory to store the results')
    parser.add_argument(
        '--mark',
        type=str,
        help='a string will be included in the filenames of results',
        default='')
    parser.add_argument('--src-mag',
                        type=int,
                        help='the max magnification of WSI the model',
                        default=40)
    parser.add_argument(
        '--mag',
        type=int,
        help='the magnification of kfb WSI the model will read on',
        default=10)
    parser.add_argument('--psize',
                        type=int,
                        help='the patch size of the model input',
                        default=256)
    parser.add_argument(
        '--laid',
        type=int,
        help=
        'a padding length will affect the output of the model we use to reduce block effect',
        default=8)
    parser.add_argument(
        '--scale',
        type=int,
        help=
        'save scale according to the mag argument, if src mag is 10, 1 for 10x, 4 for 2.5x',
        default=4)
    parser.add_argument(
        '--ncpu',
        type=int,
        help=
        'process number for multiprocessing to speed up task, file level, need to consdier gpu memory',
        default=5)
    parser.add_argument(
        '--threshold',
        type=float,
        help='the threshold to determine the positive (p>threshold)',
        default=0.5)
    parser.add_argument('--save-ori',
                        action='store_true',
                        help='save the scaled WSI')
    parser.add_argument('--save-heat',
                        action='store_true',
                        help='save the heatmap overlapped on the scaled WSI')
    parser.add_argument('--save-mask',
                        action='store_true',
                        help='save the scaled binary mask png')
    parser.add_argument('--save-contour',
                        action='store_true',
                        help='save the contours of mask as json file')
    parser.add_argument('--post',
                        action='store_true',
                        help='enable post processing to finetune mask result')
    if default:
        return parser.parse_args([])
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    start = time.time()
    target = args.src
    src_ext = args.src_ext
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(args.model, map_location=device)
    net.eval()
    label2color = {0: [0, 0, 0], 1: [227, 207, 87], 2: [178, 34, 34]}
    multiprocessing.set_start_method('spawn')
    manager = Manager()
    Tasks = manager.dict()
    #Tasks={}
    if target.endswith(src_ext):
        testOne(target, device, net, args, update_state, Tasks)
    else:
        #multiprocessing.set_start_method('spawn')
        p = Pool(args.ncpu)
        im_paths = [
            p for p in glob(f'{target}/*{src_ext}')
            if p.endswith('_mask.png') == False
        ]
        im_paths = sorted(im_paths)
        #im_paths = im_paths[:300]
        #ii = im_paths.index('/mnt/md0/_datasets/BCa_QiLu/kfb_unannotated_soft_link_2021_0219/17411.15.kfb')
        #p.starmap(testOneWithAlarm, zip(im_paths, repeat(device), repeat(net), repeat(args), repeat(update_state), repeat(Tasks)))
        for im_path in im_paths:
            p.apply_async(testOne,
                          (im_path, device, net, args, update_state, Tasks))
            #testOne(im_path, device, net, args, update_state, Tasks)
            #break
        p.close()
        p.join()
    end = time.time()
    print('time:', (end - start) / 60, 'min')
