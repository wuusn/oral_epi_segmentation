old_dir = '/mnt/md0/_datasets/OralCavity/tma@10_2/patch@256'
src_dir = '/mnt/D/patch@256'
template_dir = '/mnt/D/TMA_combine_mask_10x_corrected'

import glob
import os
import shutil

#shutil.rmtree(src_dir, ignore_errors=True)
#os.makedirs(src_dir, exist_ok=True)
#os.makedirs(f'{src_dir}/train/WU', exists_ok=True)
#os.makedirs(f'{src_dir}/val/WU', exists_ok=True)
#os.makedirs(f'{src_dir}/train/OSU', exists_ok=True)
#os.makedirs(f'{src_dir}/val/OSU', exists_ok=True)

#phrases = ['train', 'val']
#for phrase in phrases:
#    tem_dirs = glob.glob(f'{old_dir}/{phrase}/*/*')
#    for tem_dir in tem_dirs:
#        up_dir = os.path.dirname(tem_dir)
#        name = tem_dir.split('/')[-1]
#        cohort = tem_dir.split('/')[-2]
#        if cohort == 'OSU' and phrase == 'train':
#            name = name[:-2]
#        save_dir = up_dir.replace(old_dir, src_dir)
#        os.makedirs(save_dir, exist_ok=True)
#        os.symlink(tem_dir, f'{save_dir}/{name}')


paths = glob.glob(f'{template_dir}/*/*/*.jpg')

names = []
for path in paths:
    cohort = path.split('/')[-2]
    name = path.split('/')[-1].replace('.jpg', '')
    names.append(name)

#print(names)
paths = glob.glob(f'{src_dir}/*/*/*')
for path in paths:
    phrase = path.split('/')[-3]
    name = path.split('/')[-1]
    cohort = path.split('/')[-2]
    if cohort == 'OSU' and phrase == 'train':
        name = name[:-2]

    if cohort == 'OSU' and phrase == 'val':
        name = name.replace('_N', '')
    if  name not in names:
        os.system(f"rm -rf '{path}'")
        #print(path)
