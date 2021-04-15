from osu_patient_info_xls import xls2dic
from glob import glob
import os
from multiprocessing import Pool
from itertools import repeat
from PIL import Image

def task_tma2patch(src_dir, cklst, params):
    im_ext = params['im_ext']
    cpu_cores = params['cpu_cores']
    params['cklst'] = cklst

    im_paths = glob(f'{src_dir}/*.{im_ext}')

    p = Pool(cpu_cores//2)
    p.starmap(one_tma2patch, zip(im_paths, repeat(params)))

def one_tma2patch(im_path, params):
    im_ext = params['im_ext']
    mask_ext = params['mask_ext']
    cklst = params['cklst']
    tar_dir = params['tar_dir']
    src_mag = params['src_mag']
    tar_mag = params['tar_mag']
    patch_size = params['patch_size']

    name = im_path.split('/')[-1].replace(f'.{im_ext}', '')
    mask_path = im_path.replace(f'.{im_ext}', f'.{mask_ext}')
    if os.path.exists(mask_path) == False:
        mask_path = im_path.replace(f'.{im_ext}', f'_mask.{mask_ext}')
    label = 'tumor'
    if cklst!=None:
        key = name.replace('oral_cavit', '')
        key = key.replace('_', '-')
        label = 'nontumor' if cklst[key][1]=='N' else label
    output_dir = f'{tar_dir}/{tar_mag}x{patch_size}/{label}/{name}'
    os.makedirs(output_dir, exist_ok=True)

    im = Image.open(im_path)
    scale = src_mag//tar_mag
    w,h = im.size
    im = im.resize((w//scale, h//scale), Image.BICUBIC)
    w,h = im.size
    mask = Image.open(mask_path)
    mask = mask.resize((w,h), Image.NEAREST)

    psize = patch_size
    for j in range(0,h,patch_size):
        for i in range(0, w, patch_size):
            patch = im.crop((i,j,i+psize,j+psize))
            pmask = mask.crop((i,j,i+psize,j+psize))
            patch.save(f'{output_dir}/{name}_{i}_{j}.png')
            pmask.save(f'{output_dir}/{name}_{i}_{j}_mask.png')







if __name__ == '__main__':
    xls_path = '/mnt/md0/_datasets/OralCavity/TMA_arranged/OSU/OSU_Oral_Cavity_TMA_Maps.xlsx'
    id2P = xls2dic(xls_path) #{im_id: [patient_id,(non)tumor]}

    wu_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/masked' 
    osu_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/OSU/masked' 
    tar_dir = '/mnt/md0/_datasets/OralCavity/tma'

    params=dict(
            tar_dir=tar_dir,
            src_mag=40,
            mask_ext = 'png',
            cpu_cores = 40,
        )


    for tar_mag in [20,10,5]:
        for patch_size in [1024,768,512,256,128,64]:
            params['tar_mag'] = tar_mag
            params['patch_size'] = patch_size
            params['im_ext'] = 'jpg'
            task_tma2patch(osu_dir, id2P, params)
            params['im_ext'] = 'tif'
            task_tma2patch(wu_dir, None, params)

            
