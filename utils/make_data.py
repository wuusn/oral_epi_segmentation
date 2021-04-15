from glob import glob
from os import link
from PIL import Image
from os.path import exists
from os import makedirs
from multiprocessing import Pool
from time import time
from sklearn.model_selection import train_test_split
from itertools import repeat

src = '/mnt/md0/_datasets/OralCavity/TMA_arranged'
target = '/mnt/md0/_datasets/OralCavity/tma@10'
downby = 4
psize=256
seed = 42

def make_train(img_path, phase):
    type = img_path.split('.')[-1]
    group = img_path.split('/')[-3]
    fname = img_path.split('/')[-1].split('.')[0]
    if type == 'jpg':
        mask_path = img_path.replace('.jpg', '.png')
        if not exists(mask_path):
            mask_path = glob(img_path.replace('.jpg', '_mask*.png'))[0]
    else:
        mask_path = img_path.replace('.tif', '.png')

    pil_im = Image.open(img_path)
    pil_mask = Image.open(mask_path)

    w,h = pil_im.size
    pil_im = pil_im.resize((w//downby,h//downby), Image.BICUBIC)
    pil_mask =pil_mask.resize((w//downby,h//downby), Image.NEAREST)
    w,h = pil_im.size

    dir=f'{target}/masked@10/{phase}/{group}'
    makedirs(dir, exist_ok=True)
    pil_im.save(f'{dir}/{fname}.jpg')
    pil_mask.save(f'{dir}/{fname}_mask.png')

    dir = f'{target}/patch@{psize}/{phase}/{group}/{fname}'
    makedirs(dir, exist_ok=True)

    for j in range(0,h,psize):
        for i in range(0,w,psize):
            patch = pil_im.crop((i,j,i+psize,j+psize))
            pmask = pil_mask.crop((i,j,i+psize,j+psize))
            patch.save(f'{dir}/{fname}_{i}_{j}.png')
            pmask.save(f'{dir}/{fname}_{i}_{j}_mask.png')

def make_test(img_path):
    dir = f'{target}/nomask@10/OSU'
    fname = img_path.split('/')[-1].split('.')[0]
    pil_im = Image.open(img_path)
    w,h = pil_im.size
    pil_im = pil_im.resize((w//downby,h//downby), Image.BICUBIC)
    pil_im.save(f'{dir}/{fname}.jpg')

if __name__ == '__main__':
    p = Pool(20)
    start = time()
    # masked for train
    img_paths = glob(f'{src}/*/masked/*.jpg')
    img_paths.extend(glob(f'{src}/*/masked/*.tif'))
    train_paths, val_paths = train_test_split(img_paths, test_size=int(len(img_paths)/10), random_state=seed)
    p.starmap(make_train, zip(train_paths, repeat('train')))
    p.starmap(make_train, zip(val_paths, repeat('val')))

    # unmask for test
    img_paths = glob(f'{src}/OSU/nomask/*.jpg')
    dir = f'{target}/nomask@10/OSU'
    makedirs(dir, exist_ok=True)
    p.map(make_test, img_paths)
    end = time()
    print(f'done in {(end-start)//60} min')




