import os
import numpy as np
import cv2
import glob
from utils.extract_anno import return_box, return_box_with_excluded_mask
from testWSI import get_args
from multiprocessing import Pool
from itertools import repeat
base_src = '/mnt/md0/_datasets/OralCavity/WSI' 
base_tar = '/mnt/md0/_datasets/OralCavity/WSI/extracted_tuned_masks' 
base_excluded = '/mnt/md0/_datasets/OralCavity/WSI/Oral Reanno/result' 

cohorts = [
        'SFVA', 
        'UCSF', 
        'VUMC',
    ]
c2oldc = {'SFVA': 'SFVA', 'UCSF': 'UCSF', 'VUMC': 'Vanderbilt',}
def another_one(tuned_path, excluded_src, small_src, tar_dir, args):
    name = tuned_path.split('/')[-1].replace('_tuned.png', '')
    excluded_path = f'{excluded_src}/{name}_excluded.png'
    small_src_path = f'{small_src}/{name}.png'
    anno_path = tuned_path
    if os.path.exists(excluded_path):
        box, Anno, excluded_mask = return_box_with_excluded_mask(small_src_path, anno_path, args)
        Anno = Anno*255 if np.max(Anno) < 255 else Anno
        excluded_mask = excluded_mask.astype(np.uint8)
        excluded_mask = excluded_mask*255 if np.max(excluded_mask)<255 else excluded_mask
        cv2.imwrite(f'{tar_dir}/{name}_tuned.png', Anno)
        cv2.imwrite(f'{tar_dir}/{name}_excluded.png', excluded_mask)
    else:
        box, Anno = return_box(small_src_path, anno_path, args)
        Anno = Anno*255 if np.max(Anno) < 255 else Anno
        cv2.imwrite(f'{tar_dir}/{name}_tuned.png', Anno)

for c in cohorts:
    tuned_src = f'{base_src}/{c2oldc[c]}/Masks/epi_unet_nonexpert'
    excluded_src =f'{base_excluded}/{c}_ReAnno' 
    small_src = f'{base_src}/{c2oldc[c].lower()}/small_src'
    args = get_args(True)
    if c=='UCSF':
        args.box_mode = 'bb'
    args.excluded_dir = excluded_src
    tuned_paths = glob.glob(f'{tuned_src}/*_tuned.png')
    tar_dir = f'{base_tar}/{c}'
    os.makedirs(tar_dir, exist_ok=True)
    p = Pool(20)
    p.starmap(another_one, zip(tuned_paths, repeat(excluded_src), repeat(small_src), repeat(tar_dir), repeat(args)))
