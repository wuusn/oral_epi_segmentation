import os
import png
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage.morphology import binary_dilation
import cv2
import numpy as np
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage import io, color, img_as_ubyte
import skimage

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

def readpng(mask_path='/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/SP06-2244 G6_anno.png'):
    r = png.Reader(filename=mask_path)
    rows = r.read()[2]
    l = list(rows)
    nl = np.asarray(l)
    return nl

src_mask_path = '/mnt/md0/_datasets/OralCavity/WSI/Vanderbilt/Masks/epi_unet_nonexpert/OTC-220-D_tuned.png'
nl = readpng(src_mask_path)
nl = nl/255 if np.max(nl)>1 else nl
nl = nl.astype(np.uint8)
nl = post_processing(nl)
post = Image.fromarray(nl*255)
post.save('post.png')
edge_weight = binary_dilation(nl==1, iterations=1) & ~ nl #
edge_weight = edge_weight.astype(np.uint8)
edge = Image.fromarray(edge_weight*255)
#edge = Image.fromarray(edge_weight)
edge.save('edge1.png')
