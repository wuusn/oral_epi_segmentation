from glob import glob
from multiprocessing import Pool
from itertools import repeat
import os
#from PIL import Image
import cv2
import numpy as np
def binary_imread(path):
    a = cv2.imread(path, 0)
    try:
        a = a/255 if np.max(a)>1 else a
    except:
        print('error', path)
    a = a.astype(np.uint8)
    return a
def tune_one_task(unet_mask_path, tuned_dir, small_src, dst_dir):
    unet_mask = binary_imread(unet_mask_path)
    dst_mask = np.copy(unet_mask)
    name = unet_mask_path.split('/')[-1].replace('_pred.png', '')
    small_src_path = f'{small_src}/{name}.png'
    tuned_FN_path = f'{tuned_dir}/{name}_False Negative.png'
    tuned_FP_path = f'{tuned_dir}/{name}_False Positive.png'
    tuned_excluded_path = f'{tuned_dir}/{name}_excluded.png'
    ori = cv2.imread(small_src_path)
    bg = np.any(ori<=[225,225,225], axis=-1)
    bg = bg.astype(np.uint8)
    if os.path.exists(tuned_FN_path):
        tuned_FN = binary_imread(tuned_FN_path).astype(np.bool)
        dst_mask[tuned_FN]= 1
    if os.path.exists(tuned_FP_path):
        tuned_FP = binary_imread(tuned_FP_path).astype(np.bool)
        dst_mask[tuned_FP]= 0
    if os.path.exists(tuned_excluded_path):
        tuned_excluded = binary_imread(tuned_excluded_path).astype(np.bool)
        dst_mask[tuned_excluded] = 0
    dst_mask = dst_mask&bg
    os.makedirs(dst_dir, exist_ok=True)
    cv2.imwrite(f'{dst_dir}/{name}_tuned.png', dst_mask*255)

if __name__ == '__main__':
    p = Pool(40//2)
    base_reanno='/mnt/md0/_datasets/OralCavity/WSI/Oral Reanno/result'
    base_small='/mnt/md0/_datasets/OralCavity/WSI' 
    base_unet='/mnt/md0/_datasets/OralCavity/WSI/Oral Reanno/src'
    base_dst = base_small
    phases = ['SFVA', 'UCSF', 'VUMC']
    old_phases=['SFVA', 'UCSF', 'Vanderbilt']
    small_phases=['sfva', 'ucsf', 'vanderbilt']
    for i in [1]:
        unet_mask_dir = f'{base_unet}/{phases[i]}_pred'
        tuned_dir =f'{base_reanno}/{phases[i]}_ReAnno'
        dst_dir =f'{base_dst}/{old_phases[i]}/Masks/epi_unet_nonexpert'
        small_src = f'{base_small}/{small_phases[i]}/small_src'
        unet_mask_paths = glob(f'{unet_mask_dir}/*_pred.png')
        p.starmap(tune_one_task, zip(unet_mask_paths, repeat(tuned_dir), repeat(small_src), repeat(dst_dir)))

    #test one
    #unet_mask_path = '/mnt/md0/_datasets/OralCavity/WSI/Oral Reanno/src/VUMC_pred/OTC-1-D_pred.png' 
    #tuned_dir = '/mnt/md0/_datasets/OralCavity/WSI/Oral Reanno/result/VUMC_ReAnno' 
    #dst_dir = '/tmp/mytest'
    #small_src = '/mnt/md0/_datasets/OralCavity/WSI/vanderbilt/small_src' 
    #tune_one_task(unet_mask_path, tuned_dir, small_src, dst_dir)

