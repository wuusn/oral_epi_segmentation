import glob
import os
from PIL import Image

new_size = 4096
#tar_size = 1024
tar_size = 512
ori_src = '/mnt/md0/_datasets/OralCavity/tma@10_2/patch@256'
tar_dir = f'/mnt/md0/_datasets/OralCavity/tma_3/{tar_size}'

d = {}

for path in glob.glob(f'{ori_src}/train/OSU/*'):
    name = path.split('/')[-1][:-2]
    d[name] = 'train'

for path in glob.glob(f'{ori_src}/train/WU/*'):
    name = path.split('/')[-1]
    d[name] = 'train'


for path in glob.glob(f'{ori_src}/val/OSU/*'):
    name = path.split('/')[-1]
    d[name] = 'val'


for path in glob.glob(f'{ori_src}/val/WU/*'):
    name = path.split('/')[-1]
    d[name] = 'val'

src = '/mnt/md0/_datasets/OralCavity/TMA_arranged'
im_paths = glob.glob(f'{src}/epi_tumor/*/*.jpg')
im_paths.extend(glob.glob(f'{src}/epi_nontumor/*/*.jpg'))
im_paths.extend(glob.glob(f'{src}/epi_tumor/*/*.tif'))
for im_path in im_paths:
    label = im_path.split('/')[-3]
    ext = im_path.split('.')[-1]
    name = im_path.split('/')[-1].replace(f'.{ext}', '')
    mask_path = im_path.replace(f'.{ext}', '_mask.png')
    if os.path.exists(mask_path) == False:
        mask_path = im_path.replace(f'.{ext}', '.png')
    im = Image.open(im_path)
    mask = Image.open(mask_path)
    width, height = im.size
    mask = mask.resize(im.size, Image.BICUBIC)
    
    left = (width - new_size)/2
    top = (height- new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2

    im = im.crop((left, top, right, bottom))
    mask = mask.crop((left, top, right, bottom))
    im = im.resize((tar_size,tar_size))
    mask = mask.resize((tar_size,tar_size))

    phase = d[name]
    im.save(f'{tar_dir}/{phase}/{label}/{name}.png')
    mask.save(f'{tar_dir}/{phase}/{label}/{name}_mask.png')


