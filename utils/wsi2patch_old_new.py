from glob import glob
from multiprocessing import Pool
from itertools import repeat
import os
import large_image
from PIL import Image
import cv2

def task_wsi2patch(ori_dir, mask_dir, tar_dir, params):
    src_paths = glob('%s/*.%s' % (ori_dir, params['src_ext']))

    params['mask_dir'] = mask_dir
    params['tar_dir'] = tar_dir

    p = Pool(params['cpu_cores']//2)
    p.starmap(one_wsi2patch, zip(src_paths, repeat(params)))

def one_wsi2patch(src_path, params):
    src_mag = params['src_mag']
    tar_mag = params['tar_mag']
    mask_mag = param['mask_mag']
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

    name = src_path.split('/')[-1].replace(f'.{src_ext}','')
    #label = mask_dir.split('/')[-1]
    #out_dir = f'{tar_dir}/{label}/{name}'
    out_dir = f'{tar_dir}/{name}'
    os.makedirs(out_dir, exist_ok=True)

    mask_path = f'{mask_dir}/{name}{mask_ext}'
    if os.path.exists(mask_path) == False:
        return
    mask = cv2.imread(mask_path, 0)
    scale = src_mag*patch_size//tar_mag
    small_mask = cv2.resize(mask, (Top//scale, Left//scale), Image.BICUBIC)

    tops, lefts = (small_mask >0).nonzero()
    size = patch_size*src_mag//tar_mag
    for i in range(len(tops)):
        left = lefts[i]*scale
        top = tops[i]*scale
        patch, _ = ts.getRegion(
                region = dict(left=left,top=top,width=size,height=size),
                format = large_image.tilesource.TILE_FORMAT_PIL)
        patch = patch.convert(mode='RGB')
        patch = patch.resize((patch_size, patch_size), Image.BICUBIC)
        patch.save(f'{out_dir}/{name}_{left}_{top}.{tar_ext}')







if __name__ == '__main__':
    wsi_dir = '/mnt/md0/_datasets/OralCavity/WSI/SFVA' 
    #tumor_mask_dir = '/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/tumor' 
    epi_fine_mask_dir = '/mnt/md0/_datasets/OralCavity/WSI/extracted_tuned_masks/SFVA'
    #nontumor_mask_dir = '/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/nontumor' 
    tar_dir = '/mnt/md0/_datasets/OralCavity/wsi_patch/SFVA/10x256' 

    mag = 10
    patch_size = 256

    params = dict(
            src_mag=40,
            tar_mag=10,
            mask_mag = 2.5,
            patch_size=256,
            src_ext='tif',
            tar_ext='png',
            mask_ext='_tuned.png',
            cpu_cores = 40,
            )

    task_wsi2patch(wsi_dir, epi_fine_mask_dir, tar_dir, params)
    #task_wsi2patch(wsi_dir, nontumor_mask_dir, tar_dir, params)
