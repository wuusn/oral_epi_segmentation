small_src = f'/mnt/md0/_datasets/OralCavity/WSI/ucsf/small_src' 
tuned_src = '/mnt/md0/_datasets/OralCavity/WSI/extracted_tuned_masks/UCSF'
pred_src = tuned_src #'/mnt/md0/_datasets/OralCavity/tma@10_2/save/milesialx2_cb_again/ucsf_result' 
sub_slide_src = '/mnt/md0/_datasets/OralCavity/WSI/ucsf/sub_slide' 
anno_src = '/mnt/md0/_datasets/OralCavity/WSI/UCSF/Masks' 
save_dir = '/mnt/D/Oral/whole_tuned_mask/UCSF' 


import glob
import cv2
from utils.extract_anno import *
from testWSI import get_args
import numpy as np
from PIL import Image
import os
args = get_args(True)
args.box_mode='bb'
small_paths = glob.glob(f'{small_src}/*.png')
for small_path in small_paths:
    small = cv2.imread(small_path)
    name = small_path.split('/')[-1].replace('.png', '')
    fname = f'{save_dir}/{name}_mask.png'
    if os.path.exists(fname):
        continue
    sub_anno = cv2.imread(f'{sub_slide_src}/{name}_anno.png', 0)
    pred = cv2.imread(f'{pred_src}/{name}_tuned.png', 0)
    #anno_path = f'{anno_src}/{name}_anno.png'
    anno_path = f'{anno_src}/{name}.png'
    Anno = cv2.imread(anno_path, 0)
#    if (sub_anno.shape!=pred.shape):
#        if (sub_anno.shape[0]%pred.shape[0]!=0):
    #print(name, Anno.shape, small.shape[:2], sub_anno.shape, pred.shape)
    box, _= return_box(small_path, anno_path, args)
    Left,Top,Right,Bottom = box
    left = Left//args.anno_scale
    top = Top//args.anno_scale
    right = Right//args.anno_scale
    bottom = Bottom//args.anno_scale
    P = np.zeros(Anno.shape)
    if Anno.shape == small.shape[:2]:
        if P[left:right, top:bottom].shape == pred.shape:
            #continue
            P[left:right, top:bottom] = pred
            P = P.astype(np.uint8)
            Image.fromarray(P).save(fname)
        elif (right-left)%(pred.shape[0])==0:
            pred = cv2.resize(pred, (bottom-top, right-left))
            P[left:right, top:bottom] = pred
            P = P.astype(np.uint8)
            Image.fromarray(P).save(fname)
        else:
            print(right-left, bottom-top)
            print(name, Anno.shape, small.shape[:2], sub_anno.shape, pred.shape)
    else:
    #elif sub_anno.shape[0]%pred.shape[0]==0:
        print(name, Anno.shape, small.shape[:2], sub_anno.shape, pred.shape)

