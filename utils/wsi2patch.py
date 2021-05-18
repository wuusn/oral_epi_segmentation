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

interp_method=PIL.Image.BICUBIC
mask_level = 2.5
tif_level = 40
psize=256
scale = 4
downlevel = 16
tar_mag = 10
WSI_dir = '/mnt/md0/_datasets/OralCavity/WSI'
#mask_dir = '/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA'
target = '/mnt/D/Oral/wsi_patch'
c2oldc = {'SFVA': 'SFVA', 'UCSF': 'UCSF', 'VUMC': 'Vanderbilt',}
def readpng(mask_path='/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/SP06-2244 G6_anno.png'):
    r = png.Reader(filename=mask_path)
    rows = r.read()[2]
    l = list(rows)
    nl = np.asarray(l)
    return nl

def task(mask_path='/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/SP06-2244 G6_anno.png', WSI_path='/mnt/md0/_datasets/OralCavity/WSI/OralCavity_SFVA/SP06-2244 G6.tif', cohort='default'):
    nl = readpng(mask_path=mask_path)
    ts = large_image.getTileSource(WSI_path)
    tiny_mask = cv2.resize(nl, (nl.shape[1]//(psize//scale),nl.shape[0]//(psize//scale)), interp_method)
    ys,xs = (tiny_mask > 0).nonzero()
    size = len(xs)
    name = WSI_path.split('/')[-1]
    name = name.replace('.tif', '')
    target_folder = f'{target}/{cohort}/{name}'
    os.makedirs(target_folder, exist_ok=True)
    for i in range(size):
        x = xs[i]*psize*scale
        y = ys[i]*psize*scale
        big_patch, _ = ts.getRegion(
                region = dict(left=x, top=y, width=psize*scale, height=psize*scale),
                format = large_image.tilesource.TILE_FORMAT_PIL
                )
        big_patch = big_patch.convert(mode='RGB')
        patch = big_patch.resize((psize,psize), interp_method)
        #mask = nl[ys[i]*(psize//scale):ys[i]*(psize//scale)+(psize//(tar_mag//mask_level)), xs[i]*(psize//scale):xs[i]*(psize//scale)+(psize//(tar_mag//mask_level))]
        mask = nl[ys[i]*(psize//scale):ys[i]*(psize//scale)+64, xs[i]*(psize//scale):xs[i]*(psize//scale)+64]
        mask = cv2.resize(mask, (psize,psize), interp_method)
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}.png')
        pil_mask = Image.fromarray(mask)
        pil_mask.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_mask.png')

if __name__ == '__main__':
    cohorts = [
                'SFVA',
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
    p.close()
    p.join()

    end = time.time()
    print('done:', (end-start)/60)




