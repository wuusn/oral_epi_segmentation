from evalmask import evalmask
from glob import glob
import cv2
from tune_mask_with_trained_non_experts import binary_imread
import numpy as np
import os
from multiprocessing import Pool
from itertools import repeat


def test_one_cohort(cohort, base_unet, base_tuned, base_excluded):
    c2oldc = {'SFVA': 'SFVA', 'UCSF': 'UCSF', 'VUMC': 'Vanderbilt',}
    unet_src = f'{base_unet}/{cohort}_pred'
    #unet_src = base_unet
    tuned_src = f'{base_tuned}/{c2oldc[cohort]}/Masks/epi_unet_nonexpert'
    excluded_src = f'{base_excluded}/{cohort}_ReAnno'
    unet_src_paths = glob(f'{unet_src}/*_pred.png')
    C = np.zeros((2,2))
    D = 0
    count = 0
    for src_path in unet_src_paths:
        name = src_path.split('/')[-1].replace('_pred.png', '')
        unet_mask = binary_imread(src_path)
        tuned_mask_path = f'{tuned_src}/{name}_tuned.png'
        tuned_mask = binary_imread(tuned_mask_path)
        excluded_mask_path = f'{excluded_src}/{name}_excluded.png'
        if os.path.exists(excluded_mask_path):
            excluded_mask = binary_imread(excluded_mask_path)
            excluded_mask = excluded_mask.astype(np.bool)
            unet_mask[excluded_mask] = 0
        cmatrix, evals = evalmask(tuned_mask, unet_mask)
        C = C + cmatrix
        D = D + evals['dice']
        count +=1
        v=evals
        print('%s\tPA: %.2f\tR: %.2f\tPPV:%.2f\tD: %.2f'\
                %(name.ljust(20), v['acc']*100, v['tpr']*100,v['ppv']*100, v['dice']*100))
    D = D/count
    ACC = (C/C.sum()).trace()
    TNR = C[0,0]/(C[0,0]+C[0,1])
    TPR = C[1,1]/(C[1,1]+C[1,0])
    PPV = C[1,1]/(C[1,1]+C[0,1])
    c2D = {'SFVA':'D3', 'Vanderbilt':'D4', 'UCSF':'D5'}
    print(c2d[cohort], cohort)
    print('PA: %.2f\tR: %.2f\tPPV: %.2f\tD: %.2f'%(ACC*100,TPR*100,PPV*100,D*100))

def test_one_model(base_model, base_tuned, base_excluded):
    cohorts = ['SFVA', 'UCSF', 'VUMC']
    c2oldc = {'SFVA': 'SFVA', 'UCSF': 'UCSF', 'VUMC': 'Vanderbilt',}
    for c in cohorts:
        base_unet = f'{base_model}/{c2oldc[c].lower()}_result'
        test_one_cohort(c, base_unet, base_tuned, base_excluded) 
if __name__ == '__main__':
    unet_base = '/mnt/md0/_datasets/OralCavity/WSI/Oral Reanno/src' 
    tuned_base = '/mnt/md0/_datasets/OralCavity/WSI'
    excluded_base = '/mnt/md0/_datasets/OralCavity/WSI/Oral Reanno/result' 
    cohorts = ['SFVA', 'UCSF', 'VUMC']
    models = ['5xUNet', '10xUNet', '10xUNet+HEA', '10xUNet+SE', '20xUNet']
    m2p={}
    m2p['5xUNet'] = '/mnt/md0/_datasets/OralCavity/tma/5x256/save/base_model' 
    m2p['10xUNet'] = '/mnt/md0/_datasets/OralCavity/tma@10_2/save/milesialx2_cb_again' 
    m2p['10xUNet+HEA'] = '/mnt/md0/_datasets/OralCavity/tma@10_2/save/milex2_cb_HEAug' 
    m2p['10xUNet+SE'] = '/mnt/md0/_datasets/OralCavity/tma@10_2/save/yushuai_3' 
    m2p['20xUNet'] = '/mnt/md0/_datasets/OralCavity/tma/20x256/save/base_model' 
    p = Pool(10)
    p.starmap(test_one_cohort, zip(cohorts, repeat(unet_base), repeat(tuned_base), repeat(excluded_base)))
    #model_paths=[m2p[m] for m in models][:2]
    #p.starmap(test_one_model, zip(model_paths, repeat(tuned_base), repeat(excluded_base)))

