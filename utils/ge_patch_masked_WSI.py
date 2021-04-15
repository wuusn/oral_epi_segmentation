import png
import os
import large_image
import cv2
import PIL
from PIL import Image
import numpy as np
interp_method=PIL.Image.BICUBIC
mask_level = 10
tif_level = 40
psize=256
scale = 4
downlevel = 16
WSI_dir = '/mnt/md0/_datasets/OralCavity/WSI/OralCavity_SFVA'
mask_dir = '/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA'
target = '/mnt/md0/_datasets/OralCavity/tma@10_2/wsi_patch'
def readpng(mask_path='/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/SP06-2244 G6_anno.png'):
    r = png.Reader(filename=mask_path)
    rows = r.read()[2]
    l = list(rows)
    nl = np.asarray(l)
    return nl

def task(mask_path='/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/SP06-2244 G6_anno.png', WSI_path='/mnt/md0/_datasets/OralCavity/WSI/OralCavity_SFVA/SP06-2244 G6.tif'):
    nl = readpng(mask_path=mask_path)
    ts = large_image.getTileSource(WSI_path)
    tiny_mask = cv2.resize(nl, (nl.shape[1]//(psize//scale),nl.shape[0]//(psize//scale)), interp_method)
    ys,xs = (tiny_mask > 0).nonzero()
    size = len(xs)
    name = WSI_path.split('/')[-1]
    name = name.replace('.tif', '')
    phase = mask_path.split('/')[-2]
    target_folder = f'{target}/{name}/{phase}'
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
        mask = nl[ys[i]*(psize//scale):ys[i]*(psize//scale)+64, xs[i]*(psize//scale):xs[i]*(psize//scale)+64]
        mask = cv2.resize(mask, (256,256), interp_method)
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}.png')
        pil_mask = Image.fromarray(mask)
        pil_mask.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_mask.png')

if __name__ == '__main__':
    task(WSI_path=f'{WSI_dir}/SP06-4230 A2.tif', mask_path=f'{mask_dir}/yellow/SP06-4230 A2.png')
    task(WSI_path=f'{WSI_dir}/SP06-4230 A2.tif', mask_path=f'{mask_dir}/red/SP06-4230 A2.png')
    task(WSI_path=f'{WSI_dir}/SP06-2244 G6.tif', mask_path=f'{mask_dir}/yellow/SP06-2244 G6.png')
    task(WSI_path=f'{WSI_dir}/SP06-2244 G6.tif', mask_path=f'{mask_dir}/red/SP06-2244 G6.png')
    task(WSI_path=f'{WSI_dir}/SP07-1191 F4.tif', mask_path=f'{mask_dir}/yellow/SP07-1191 F4.png')
    task(WSI_path=f'{WSI_dir}/SP07-1191 F4.tif', mask_path=f'{mask_dir}/red/SP07-1191 F4.png')




