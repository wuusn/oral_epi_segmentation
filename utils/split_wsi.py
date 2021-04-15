import random
random.seed(56)
import glob
import os
from sklearn.model_selection import train_test_split

wsi_path = '/mnt/md0/_datasets/OralCavity/wsi/10x256/sfva_epi_all'
tar_src = f'/mnt/md0/_datasets/OralCavity/wsi/10x256/train_sfva_epi_all'

wsi_dirs = glob.glob(f'{wsi_path}/*')
train_dirs, val_dirs = train_test_split(wsi_dirs, test_size=len(wsi_dirs)//10)
dirs = {'train': train_dirs, 'val': val_dirs}
for p in ['train', 'val']:
    tdirs = dirs[p]
    tar = f'{tar_src}/{p}/epi'
    os.makedirs(tar, exist_ok=True)
    for d in tdirs:
        name = d.split('/')[-1]
        #print(d, f'{tar}/{name}')
        os.symlink(d, f'{tar}/{name}')
