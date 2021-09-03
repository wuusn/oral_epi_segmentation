import time
import png
import os
import large_image
import cv2
import PIL
from PIL import Image
import numpy as np
from multiprocessing import Pool, Manager
from itertools import repeat
import glob
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage import io, color, img_as_ubyte
import skimage
from scipy.ndimage.morphology import binary_dilation
import random
import sklearn
import sklearn.model_selection

interp_method=PIL.Image.BICUBIC
mask_level = 2.5
tif_level = 40
psize=256
scale = 4
downlevel = 16
tar_mag = 10
#WSI_dir = '/mnt/md0/_datasets/OralCavity/WSI'
WSI_dir = '/mnt/disk1/Oral'
target = '/mnt/disk1/wsi_patch_OnInOut'
c2oldc = {'SFVA': 'SFVA', 'UCSF': 'UCSF', 'VUMC': 'Vanderbilt',}
def readpng(mask_path='/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/SP06-2244 G6_anno.png'):
    r = png.Reader(filename=mask_path)
    rows = r.read()[2]
    l = list(rows)
    nl = np.asarray(l)
    return nl

def post_processing(mask, scale=1):
    mask= mask / 255 if np.max(mask)>1 else mask
    mask = mask.astype(np.bool)
    area_thresh = 200//scale
    mask_opened = remove_small_objects(mask, min_size=area_thresh)
    mask_removed_area = ~mask_opened & mask
    mask = mask_opened > 0

    min_size = 300//scale
    img_reduced = skimage.morphology.remove_small_holes(mask, area_threshold=min_size)
    img_small = img_reduced & np.invert(mask)
    mask = img_reduced
    mask = mask.astype(np.uint8)
    kernel = np.ones((5,5), dtype=np.uint8)
    new = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return new

def task(mask_path='/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/SP06-2244 G6_anno.png', WSI_path='/mnt/md0/_datasets/OralCavity/WSI/OralCavity_SFVA/SP06-2244 G6.tif', cohort='default'):
    nl = readpng(mask_path=mask_path)
    ts = large_image.getTileSource(WSI_path)
    #tiny_mask = cv2.resize(nl, (nl.shape[1]//(psize//scale),nl.shape[0]//(psize//scale)), interp_method)
    post = post_processing(nl)
    nl2 = nl/255 if np.max(nl)>1 else nl
    nl2 = nl2.astype(np.uint8)

    # On
    edge = binary_dilation(nl2==1, iterations=1) & ~nl2
    #ys,xs = (tiny_mask > 0).nonzero() #
    ys,xs = (edge > 0).nonzero() #
    size = len(xs)
    idxs = list(range(0, size))
    sel_size = size//64//3
    _, sel = sklearn.model_selection.train_test_split(idxs, test_size=sel_size, random_state=56)
    name = WSI_path.split('/')[-1]
    name = name.replace('.tif', '')
    print(name)
    target_folder = f'{target}/{cohort}/{name}'
    os.makedirs(target_folder, exist_ok=True)
    for i in sel:
        x = xs[i]*downlevel-psize*scale//2#psize*scale
        y = ys[i]*downlevel-psize*scale//2#psize*scale
        if x < 0 or y <0:
            continue
        big_patch, _ = ts.getRegion(
                region = dict(left=x, top=y, width=psize*scale, height=psize*scale),
                format = large_image.tilesource.TILE_FORMAT_PIL
                )
        big_patch = big_patch.convert(mode='RGB')
        patch = big_patch.resize((psize,psize), interp_method)
        #mask = nl[ys[i]*(psize//scale):ys[i]*(psize//scale)+(psize//(tar_mag//mask_level)), xs[i]*(psize//scale):xs[i]*(psize//scale)+(psize//(tar_mag//mask_level))]
        #mask = nl[ys[i]*(psize//scale):ys[i]*(psize//scale)+64, xs[i]*(psize//scale):xs[i]*(psize//scale)+64]
        stride = psize//scale//2
        mask = nl[ys[i]-stride:ys[i]+stride, xs[i]-stride:xs[i]+stride]
        if np.sum(mask) < (psize*psize/scale/scale/4):
            continue
        mask = cv2.resize(mask, (psize,psize), interp_method)
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_on.png')
        pil_mask = Image.fromarray(mask)
        pil_mask.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_on_mask.png')

    # out
    out = ~binary_dilation(nl2==1, iterations=stride) & ~ nl2 #
    out = out.astype(np.uint8)
    tiny_out = cv2.resize(out, (out.shape[1]//(psize//scale),out.shape[0]//(psize//scale)), interp_method)
    ys,xs = (tiny_out > 0).nonzero() #
    size = len(xs)
    idxs = list(range(0, size))
    if size < sel_size:
        sel = list(range(0, size))
    else:
        _, sel = sklearn.model_selection.train_test_split(idxs, test_size=sel_size, random_state=56)
    for i in sel:
        x = xs[i]*psize*scale
        y = ys[i]*psize*scale
        if x < 0 or y <0:
            continue
        big_patch, _ = ts.getRegion(
                region = dict(left=x, top=y, width=psize*scale, height=psize*scale),
                format = large_image.tilesource.TILE_FORMAT_PIL
                )
        big_patch = big_patch.convert(mode='RGB')
        patch = big_patch.resize((psize,psize), interp_method)
        mask = nl[ys[i]*(psize//scale):ys[i]*(psize//scale)+64, xs[i]*(psize//scale):xs[i]*(psize//scale)+64]
        mask = cv2.resize(mask, (psize,psize), interp_method)
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_out.png')
        pil_mask = Image.fromarray(mask)
        pil_mask.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_out_mask.png')

    # in , out = in
    out = binary_dilation(nl2==1, iterations=stride) & nl2 #
    out = out.astype(np.uint8)
    tiny_out = cv2.resize(out, (out.shape[1]//(psize//scale),out.shape[0]//(psize//scale)), interp_method)
    ys,xs = (tiny_out > 0).nonzero() #
    size = len(xs)
    idxs = list(range(0, size))
    if size < sel_size:
        sel = list(range(0, size))
    else:
        _, sel = sklearn.model_selection.train_test_split(idxs, test_size=sel_size, random_state=56)
    for i in sel:
        x = xs[i]*psize*scale
        y = ys[i]*psize*scale
        if x < 0 or y <0:
            continue
        big_patch, _ = ts.getRegion(
                region = dict(left=x, top=y, width=psize*scale, height=psize*scale),
                format = large_image.tilesource.TILE_FORMAT_PIL
                )
        big_patch = big_patch.convert(mode='RGB')
        patch = big_patch.resize((psize,psize), interp_method)
        mask = nl[ys[i]*(psize//scale):ys[i]*(psize//scale)+64, xs[i]*(psize//scale):xs[i]*(psize//scale)+64]
        if np.sum(mask) < (psize*psize/scale/scale/4):
            continue
        mask = cv2.resize(mask, (psize,psize), interp_method)
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_in.png')
        pil_mask = Image.fromarray(mask)
        pil_mask.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_in_mask.png')
if __name__ == '__main__':
    cohorts = [
                #'SFVA',
                'UCSF',
                'VUMC',
            ]
    start = time.time()
    #task(WSI_path=f'{WSI_dir}/SP06-4230 A2.tif', mask_path=f'{mask_dir}/yellow/SP06-4230 A2.png')
    #task(WSI_path=f'{WSI_dir}/SP06-4230 A2.tif', mask_path=f'{mask_dir}/red/SP06-4230 A2.png')
    #task(WSI_path=f'{WSI_dir}/SP06-2244 G6.tif', mask_path=f'{mask_dir}/yellow/SP06-2244 G6.png')
    #task(WSI_path=f'{WSI_dir}/SP06-2244 G6.tif', mask_path=f'{mask_dir}/red/SP06-2244 G6.png')
    #task(WSI_path=f'{WSI_dir}/SP07-1191 F4.tif', mask_path=f'{mask_dir}/yellow/SP07-1191 F4.png')
    #task(WSI_path=f'{WSI_dir}/SP07-1191 F4.tif', mask_path=f'{mask_dir}/red/SP07-1191 F4.png')
    #task(WSI_path=f'{WSI_dir}/SP06-1112 D3.tif', mask_path=f'{mask_dir}/SP06-1112 D3_mask.png')
    p = Pool(20)
    for cohort in cohorts:
        cohort_wsi_dir = f'{WSI_dir}/{c2oldc[cohort]}'
        mask_dir = f'{cohort_wsi_dir}/Masks/epi_unet_nonexpert'
        wsi_paths = glob.glob(f'{cohort_wsi_dir}/*.tif')
        #wsi_paths = wsi_paths[:1]
        for wsi_path in wsi_paths:
            name = wsi_path.split('/')[-1]
            name = name.replace('.tif', '')
            mask_path = f'{mask_dir}/{name}_tuned.png'
            p.apply_async(task, (mask_path, wsi_path, cohort))
            #task(mask_path, wsi_path, cohort)
            #break
    p.close()
    p.join()

    end = time.time()
    print('done:', (end-start)/60)




