from glob import glob
import numpy as np
from multiprocessing import Pool
from itertools import repeat
import os
import large_image
from PIL import Image
import cv2
from utils.extract_anno import return_box, return_box_with_excluded_mask
from testWSI import get_args
import time

args = get_args(True)
def task_wsi2patch(ori_dir, mask_dir, tar_dir, params):
    src_paths = glob('%s/*%s' % (ori_dir, params['src_ext']))

    params['mask_dir'] = mask_dir
    params['tar_dir'] = tar_dir

    #one_wsi2patch('/mnt/md0/_datasets/OralCavity/WSI/SFVA/SP06-2244 G6.tif', params)
    p = Pool(params['cpu_cores']//2)
    p.starmap(one_wsi2patch, zip(src_paths, repeat(params)))

def one_wsi2patch(src_path, params):
    src_mag = params['src_mag']
    tar_mag = params['tar_mag']
    patch_size = params['patch_size']
    src_ext = params['src_ext']
    tar_ext = params['tar_ext']
    mask_ext = params['mask_ext']
    tar_dir = params['tar_dir']
    mask_dir = params['mask_dir']

    try:
        ts = large_image.getTileSource(src_path)
        Left = ts.sizeY
        Top = ts.sizeX
    except Exception as e:
        print(e)
        return

    name = src_path.split('/')[-1].replace(f'{src_ext}','')
    cohort = mask_dir.split('/')[-3].lower()
    out_dir = f'{tar_dir}/{name}'

    mask_path = f'{mask_dir}/{name}{mask_ext}'
    small_src_path = f'/mnt/md0/_datasets/OralCavity/WSI/{cohort}/small_src/{name}.png'
    if os.path.exists(small_src_path) == False: return
    os.makedirs(out_dir, exist_ok=True)
    box, _ = return_box(small_src_path, mask_path, args)
    left,top,right,bottom = [c//16 for c in box]
    if os.path.exists(mask_path) == False:
        return
    mask = cv2.imread(mask_path, 0)
    for i in range(top,bottom,patch_size//4):
        for j in range(left,right,patch_size//4):
            patch, _ = ts.getRegion(
                    region = dict(left=i*16,top=j*16,width=patch_size*4,height=patch_size*4),
                    format = large_image.tilesource.TILE_FORMAT_PIL)
            patch = patch.convert(mode='RGB')
            patch = patch.resize((patch_size, patch_size), Image.BICUBIC)
            crop_mask = mask[j:j+patch_size//4,i:i+patch_size//4]
            crop_mask = cv2.resize(crop_mask, (patch_size,patch_size))
            ratio = np.sum(crop_mask)/255/(patch_size**2)
            #if ratio<.2 or ratio>.8: continue
            patch.save(f'{out_dir}/{name}_{i*16}_{j*16}{tar_ext}')
            cv2.imwrite(f'{out_dir}/{name}_{i*16}_{j*16}_mask{tar_ext}', crop_mask)







if __name__ == '__main__':
    start = time.time()
    wsi_dir = '/mnt/md0/_datasets/OralCavity/WSI/SFVA' 
    tumor_mask_dir = '/mnt/md0/_datasets/OralCavity/WSI/SFVA/Masks/epi_unet_nonexpert' 
    #nontumor_mask_dir = '/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/nontumor' 
    tar_dir = '/mnt/md0/_datasets/OralCavity/wsi/10x256/sfva_epi_all' 

    mag = 10
    patch_size = 256

    params = dict(
            src_mag=40,
            tar_mag=10,
            patch_size=256,
            src_ext='.tif',
            tar_ext='.png',
            mask_ext='_tuned.png',
            cpu_cores = 40,
            )

    task_wsi2patch(wsi_dir, tumor_mask_dir, tar_dir, params)
    #task_wsi2patch(wsi_dir, nontumor_mask_dir, tar_dir, params)
    end = time.time()
    print('time:', (end-start)/60)
