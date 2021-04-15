from PIL import Image
import os
import numpy as np
import cv2
from .tune_mask_with_trained_non_experts import binary_imread

def get_boxes(mom, scale):
    np_mom = np.array(mom)
    bg = np.any(np_mom != [235,235,235], axis=-1)
    bg = bg.astype(np.uint8)
    small_bg = cv2.resize(bg, (bg.shape[1]//scale, bg.shape[0]//scale))
    kernel = np.ones((3,3), dtype=np.uint8)
    small_bg = cv2.morphologyEx(small_bg, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(small_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def check_contour(cs):
    new =[]
    if len(cs)==4:
        for c in cs:
            c=c[0]
            new.append(c)
        return new
    last=None
    for c in cs:
        c=c[0]
        if len(new)==0:
            new.append(c)
            last=c
        elif len(new)==4:
            return new
        else:
            d = abs(c[0]-last[0])+abs(c[1]-last[1])
            if d>10:
                new.append(c)
                last=c
            elif(len(new)!=1):
                new.pop()
                #new.append(c)
                #last=c
    return new 

def select_box(anno, boxes):
    selects=[]
    for box in boxes:
        #a,b,c,d = check_contour(box)
        a,b,c,d = box
        a=a[0]
        b=b[0]
        c=c[0]
        d=d[0]
        if(np.sum(anno[a[1]:b[1], a[0]:d[0]])>1):
            selects.append([a,b,c,d])
    if(len(selects)>1):
        return None
    else:
        return selects[0]
def get_boundingBox(anno):
    x,y,w,h = cv2.boundingRect(anno)
    a = (x,y)
    b = (x,y+h)
    c = (x+w,y+h)
    d = (x+w,y)
    return [a,b,c,d]

def return_box(src_path, mask_path, args, scale=40):
    im = Image.open(src_path)
    anno_path = mask_path
    anno = cv2.imread(anno_path, 0)
    small_anno = cv2.resize(anno, (im.size[0]//scale, im.size[1]//scale))
    if args.box_mode == 'bb':
        #name = src_path.split('/')[-1].replace('.png', '')
        #anno_coarse_path = f'/mnt/md0/_datasets/OralCavity/WSI/UCSF/Masks/{name}.png' 
        #anno_coarse = cv2.imread(anno_coarse_path, 0) 
        #small_anno_coarse = cv2.resize(anno_coarse, (im.size[0]//scale, im.size[1]//scale))
        small_anno_coarse = small_anno
        select = get_boundingBox(small_anno_coarse)
    elif args.box_mode == 'full':
        select = None
    else:
        boxes = get_boxes(im, scale)
        select = select_box(small_anno, boxes)
    if select == None:
        w,h=(anno.shape[1],anno.shape[0])
        left=0
        top=0
        right = h
        bottom =w
        Left=left
        Top=top
        Right=h*args.anno_scale
        Bottom=w*args.anno_scale
    else:
        tmp_scale = anno.shape[1]//im.size[0]*scale
        left = select[0][1] * tmp_scale
        left = int(left)
        top = select[0][0] * tmp_scale
        top = int(top)
        right = select[2][1] * tmp_scale
        right = int(right)
        bottom = select[2][0] * tmp_scale
        bottom = int(bottom)

        Left = left * args.anno_scale
        Top = top * args.anno_scale
        Right = right * args.anno_scale
        Bottom = bottom * args.anno_scale

    anno = anno[left:right,top:bottom]
    return (Left,Top,Right,Bottom), anno

def return_box_with_excluded_mask(src_path, mask_path, args, scale=40):
    (Left,Top, Right, Bottom), anno = return_box(src_path, mask_path, args, scale)
    left = Left//args.anno_scale
    top = Top//args.anno_scale
    right = Right //args.anno_scale
    bottom = Bottom//args.anno_scale
    name = src_path.split('/')[-1].replace('.png', '')
    excluded_mask_path = f'{args.excluded_dir}/{name}_excluded.png'
    bool_mask = np.zeros((right-left, bottom-top))
    if os.path.exists(excluded_mask_path):
        bool_mask = binary_imread(excluded_mask_path)
        bool_mask = bool_mask[left:right,top:bottom]
    #bool_mask = bool_mask.astype(np.bool)
    return (Left, Top, Right, Bottom), anno, bool_mask

if __name__ == '__main__':
    mom_path = '/mnt/md0/_datasets/OralCavity/WSI/downby16/SFVA/SP08-4879 A1.png' 
    anno_path = '/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/tumor/SP08-4879 A1.png' 
    #boxes = get_boxes(mom_path)
    #select = select_box(anno_path, boxes)
    box, anno = return_box(mom_path, anno_path)

    print(box)




