import glob
import os
from sklearn.model_selection import train_test_split
seed = 57

def split(src, save_dir):
    os.makedirs(save_dir + '/train', exist_ok=True)
    os.makedirs(save_dir + '/val', exist_ok=True)
    wsis = glob.glob(f'{src}/*')
    size = int(len(wsis)/10)
    train_wsis, val_wsis = train_test_split(wsis, test_size=size, random_state=seed)
    phrases = {'train': train_wsis, 'val': val_wsis}
    for phrase in ['train', 'val']:
        t_wsis = phrases[phrase]
        for wsi in t_wsis:
            name = wsi.split('/')[-1]
            os.symlink(wsi, f'{save_dir}/{phrase}/{name}')

if __name__ == '__main__':
    #src_dir = '/mnt/D/Oral/wsi_patch_edge'
    src_dir = '/mnt/D/Oral/wsi_patch_OnInOut'
    tar_dir = '/mnt/D/Oral/train_wsi_OnInOut'
    cohort = [
            'SFVA', 
            #'UCSF', 
            #'VUMC',
            ]
    for c in cohort:
        split(f'{src_dir}/{c}', f'{tar_dir}/{c}')
