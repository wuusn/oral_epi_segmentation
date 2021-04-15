from utils.evalmask import evalmask
import math
from glob import glob
import cv2
from utils.tune_mask_with_trained_non_experts import binary_imread
import numpy as np
import os
from multiprocessing import Pool, Manager
from itertools import repeat
#from utils.extract_anno import return_box, return_box_with_excluded_mask
import cv2
from testWSI import get_args
import time
import xlsxwriter

def test_one_img(src_path, tuned_src, excluded_src, args, sharedL):
    name = src_path.split('/')[-1].replace('_pred.png', '')
    unet_mask = binary_imread(src_path)
    tuned_mask_path = f'{tuned_src}/{name}_tuned.png'
    tuned_mask = binary_imread(tuned_mask_path)
    excluded_mask_path = f'{excluded_src}/{name}_excluded.png'
    Anno = tuned_mask
    if os.path.exists(excluded_mask_path):
        #box, Anno, excluded_mask = return_box_with_excluded_mask(small_src_path, anno_path, args)
        excluded_mask = binary_imread(excluded_mask_path)
        if args.net_mag != 20:
            Anno = cv2.resize(Anno, (unet_mask.shape[1], unet_mask.shape[0]))
            excluded_mask = cv2.resize(excluded_mask, (unet_mask.shape[1], unet_mask.shape[0]))
        else:
            unet_mask = cv2.resize(unet_mask, (Anno.shape[1], Anno.shape[0]))

        excluded_mask = excluded_mask.astype(np.bool)
        unet_mask[excluded_mask] = 0
    else:
        #box, Anno = return_box(small_src_path, anno_path, args)
        if args.net_mag != 20:
            Anno = cv2.resize(Anno, (unet_mask.shape[1], unet_mask.shape[0]))
        else:
            unet_mask = cv2.resize(unet_mask, (Anno.shape[1], Anno.shape[0]))

    tuned_mask = Anno

    cmatrix, evals = evalmask(tuned_mask, unet_mask)
    #C = C + cmatrix
    #D = D + evals['dice']
    #count +=1
    v=evals
    pa = v['acc']*100
    rr = v['tpr']*100
    ppv = v['ppv']*100
    dc =v['dice']*100
    sharedL.append([name,pa,rr,ppv,dc])
    #print('%s\tPA: %.2f\tR: %.2f\tPPV:%.2f\tD: %.2f'\
    #        %(name.ljust(20), v['acc']*100, v['tpr']*100,v['ppv']*100, v['dice']*100))


def test_one_cohort(args,model, cohort, base_unet, base_tuned, base_excluded, base_save_xls):
    #c2oldc = {'SFVA': 'SFVA', 'UCSF': 'UCSF', 'VUMC': 'Vanderbilt',}
    #unet_src = f'{base_unet}/{cohort}_pred'
    unet_src = base_unet
    tuned_src = f'{base_tuned}/{cohort}'#/Masks/epi_unet_nonexpert'
    excluded_src = tuned_src#f'{base_excluded}/{cohort}_ReAnno'
    unet_src_paths = sorted(glob(f'{unet_src}/*_pred.png'))
    DC=0
    PPV=0
    RR=0
    PA=0
    count = 0
    #for src_path in unet_src_paths:
    manager = Manager()
    sharedL = manager.list()
    sharedL.append(['Name', 'PA %', 'RR %', 'PPV %', 'DC %'])
    p = Pool(30)
    p.starmap(test_one_img, zip(unet_src_paths, repeat(tuned_src), repeat(excluded_src), repeat(args), repeat(sharedL)))
    excel_dir = f'{base_save_xls}/{model}' 
    os.makedirs(excel_dir, exist_ok = True)
    workbook = xlsxwriter.Workbook(f'{excel_dir}/{model}_{cohort}.xlsx')
    worksheet = workbook.add_worksheet(f'{model}_{cohort}')
    worksheet.write_row(0,0, sharedL[0])
    for i in range(1, len(sharedL)):
        name,pa,rr,ppv,dc=sharedL[i]
        if math.isnan(rr):
            print(name)
            continue
        worksheet.write_row(i,0,sharedL[i])
        count = count +1
        PA += pa
        RR += rr
        PPV += ppv
        DC +=dc
    PA = PA/count
    RR = RR/count
    PPV = PPV/count
    DC = DC/count
    worksheet.write_row(i+1,0, ['AVG', PA,RR,PPV,DC])
    workbook.close()
    c2D = {'SFVA':'D3', 'VUMC':'D4', 'UCSF':'D5'}
    print(model, cohort, c2D[cohort])
    print('PA: %.2f\tR: %.2f\tPPV: %.2f\tD: %.2f'%(PA,RR,PPV,DC))
    return model, cohort, sharedL

def test_one_model(model, cohorts, base_tuned, base_excluded, base_save_xls):
    c2oldc = {'SFVA': 'SFVA', 'UCSF': 'UCSF', 'VUMC': 'Vanderbilt',}
    m2p={}
    m2p['5xUNet'] = '/mnt/md0/_datasets/OralCavity/tma/5x256/save/base_model_epi' 
    m2p['10xUNet'] = '/mnt/md0/_datasets/OralCavity/tma@10_2/save/milesialx2_cb_again' 
    m2p['10xUNet+HEA'] = '/mnt/md0/_datasets/OralCavity/tma@10_2/save/milex2_cb_HEAug' 
    m2p['10xUNet+HEA2'] = '/mnt/md0/_datasets/OralCavity/tma/10x256/save/milex2_cb_HEAug_2'  
    m2p['10xUNet+SE'] = '/mnt/md0/_datasets/OralCavity/tma@10_2/save/yushuai_3' 
    m2p['10xUNet+SE2'] = '/mnt/md0/_datasets/OralCavity/tma/10x256/save/yushuai_4' 
    m2p['20xUNet'] = '/mnt/md0/_datasets/OralCavity/tma/20x256/save/base_model_epi_50' 

    base_model = m2p[model]
    D = {}
    for c in cohorts:
        args = get_args(True)
        mag = model.split('x')[0]
        mag = int(mag)
        #if c=='UCSF':
        #    args.box_mode == 'bb'
        args.net_mag = mag
        base_unet = f'{base_model}/{c2oldc[c].lower()}_result'
        _, _, sharedL = test_one_cohort(args, model, c, base_unet, base_tuned, base_excluded, base_save_xls) 
        D[c] = sharedL
    return D
if __name__ == '__main__':
    start = time.time()
    tuned_base = '/mnt/md0/_datasets/OralCavity/WSI/extracted_tuned_masks' 
    base_save_xls = '/mnt/md0/_datasets/OralCavity/WSI/excel_results' 
    excluded_base = tuned_base
    models = [
        #'5xUNet',
        #'10xUNet',
        #'10xUNet+HEA',
        #'10xUNet+HEA2',
        #'10xUNet+SE',
        #'10xUNet+SE2',
        '20xUNet',
    ]
    cohorts = [
        #'SFVA',
        'UCSF',
        #'VUMC',
    ]
    xls_dir = f'{base_save_xls}/slide_wise'
    os.makedirs(xls_dir, exist_ok=True)
    bigD = {}
    for model in models:
        D = test_one_model(model,cohorts, tuned_base, excluded_base, base_save_xls)
    end = time.time()
    print('time:', (end-start)/60)

