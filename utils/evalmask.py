from sklearn.metrics import confusion_matrix
import time
import os
import numpy as np
from PIL import Image
import glob

def Dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch!")

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum()+im2.sum())

def evalmask(m1,m2):
    # m1 true, m2 pred
    m1 = m1/255 if np.max(m1) > 1 else m1
    m2 = m2/255 if np.max(m2) > 1 else m2
    m1 = m1.astype(np.uint8)
    m2 = m2.astype(np.uint8)
    flat1 = m1.flatten()
    flat2 = m2.flatten()

    dice = Dice(m1,m2)
    cmatrix = confusion_matrix(flat1, flat2, labels=range(2))
    acc = (cmatrix/cmatrix.sum()).trace()
    tnr = cmatrix[0,0]/(cmatrix[0,0]+cmatrix[0,1])
    tpr = cmatrix[1,1]/(cmatrix[1,1]+cmatrix[1,0])
    ppv = cmatrix[1,1]/(cmatrix[1,1]+cmatrix[0,1])

    evals = dict(acc=acc,
                 tnr=tnr,
                 tpr=tpr,
                 ppv=ppv,
                 dice=dice)


    return cmatrix, evals


if __name__ == '__main__':
    start = time.time()
    mark = 'milesial_cmarker_30'
    #dir1 = '/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA/red'
    dir1 = '/mnt/md0/_datasets/OralCavity/WSI/Masks_SFVA'
    dir2  = f'/mnt/md0/_datasets/OralCavity/WSI/downby16/SFVA/{mark}'
    #dir2  = '/mnt/md0/_datasets/OralCavity/WSI/downby16/SFVA/one_part'
    #mark = 'pred_milesialx2_cb_17'
    C = np.zeros((2,2))
    D = 0
    E = {}
    #m2_paths = glob.glob(f'{dir2}/*_{mark}.png')
    m2_paths = glob.glob(f'{dir2}/*_{mark}.png')
    count=0
    for m2_path in m2_paths:
        name = m2_path.split('/')[-1].replace(f'_pred_{mark}', '_anno')
        m1_path = f'{dir1}/{name}'
        if(os.path.exists(m1_path)==False):
            continue
        im_m1 = Image.open(m1_path).convert(mode='L')
        im_m2 = Image.open(m2_path).convert(mode='L')
        m1 = (np.array(im_m1)/255).astype(np.uint8)
        m2 = (np.array(im_m2)/255).astype(np.uint8)
        cmatrix, evals = evalmask(m1,m2)
        E[name] = evals
        C = C + cmatrix
        D = D + evals['dice']
        count += 1
    D = D/count
    ACC = (C/C.sum()).trace()
    TNR = C[0,0]/(C[0,0]+C[0,1])
    TPR = C[1,1]/(C[1,1]+C[1,0])

    for k,v in E.items():
        print(k)
        print('acc: %.2f\ttnr: %.2f\ttpr: %.2f\tdice: %.2f'%(v['acc'], v['tnr'], v['tpr'], v['dice']))

    print()
    print('After All')
    print('ACC: %.2f\tTNR: %.2f\tTPR: %.2f\tDice: %.2f'%(ACC,TNR,TPR,D))
    end = time.time()
    print('time:', (end-start)/60)


