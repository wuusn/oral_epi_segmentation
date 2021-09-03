from unet import UNet
import torch
from glob import glob
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn
import sys
import large_image
import time
import png
import os
from matplotlib import cm
from utils.dataset import *
from torch.utils.data import DataLoader
from utils.extract_anno import *
from utils.evalmask import evalmask
import argparse

def check_result(src_path, args):
    #return False # small src quick
    name = src_path.split('/')[-1].replace(f'.{args.src_ext}', '')
    tmp_name = f'{args.save_result_dir}/{name}_pred.{args.tar_ext}'
    return os.path.exists(tmp_name)

def test_wsi_dataset(dataset, args):
    loader = DataLoader(dataset, batch_size=args.batch_size,\
                shuffle=False, num_workers=args.cpus//2, pin_memory=True)
    h = dataset.h
    w = dataset.w
    scale = dataset.scale
    downby = args.tar_scale
    psize = args.patch_size
    laid = args.laid

    #print(h,scale,downby,psize,w)
    Heat = np.zeros(\
            (h//scale//downby +psize,w//scale//downby +psize, 3)\
        ).astype(np.uint8)
    Ori = np.zeros(\
            (h//scale//downby +psize,w//scale//downby +psize, 3)\
        ).astype(np.uint8)
    Mask = np.zeros(\
            (h//scale//downby +psize,w//scale//downby +psize)\
        ).astype(np.uint8)

    I = 0
    for batch in loader:
        imgs = batch['image']
        imgs = imgs.to(device=device, dtype=torch.float32)
        output = net(imgs)
        # flip and avg
        #x1=imgs
        #y1 = output
        #y2 = net(torch.flip(x1, [-1]))
        #y2 = torch.flip(y2, [-1])
        #y3 = net(torch.flip(x1, [-2]))
        #y3 = torch.flip(y3, [-1])
        #y4 = net(torch.flip(x1, [-1, -2]))
        #y4 = torch.flip(y4, [-1, -2])
        #Y = (y1+y2+y3+y4)/4
        #Y = torch.min(y1,y2)
        #Y = torch.min(Y,y3)
        #Y = torch.min(Y,y4)
        #output = Y
        #
        output = output.squeeze(1)
        output = output[:,laid:psize+laid, laid:psize+laid]
        if net.n_classes > 1:
            output = torch.sigmoid(output)
            output = output.detach().cpu().numpy()
            pred = output[:,2,:,:]
            mask_pred = (pred>.5).astype(np.uint8) # b,h,w

        else:
            output = torch.sigmoid(output)
            pred = output.detach().cpu().numpy()
            mask_pred = (pred>.5).astype(np.uint8) # b,h,w
        
        nb = pred.shape[0]
        for i in range(nb):
            #break # small src quick
            jj = I//math.ceil(h/(psize)/scale)
            ii = I%math.ceil(h/(psize)/scale)
            top = jj * (psize) * scale
            left = ii * (psize) * scale
            jet_pred = cm.jet(pred[i])[:,:,:3]*255
            jet_pred = jet_pred.astype(np.uint8)
            bool_mask = pred[i]>.5
            img = batch['ori'][i]
            heat = np.copy(img)
            heat = heat[laid:psize+laid, laid:psize+laid, :]
            heat[bool_mask] = jet_pred[bool_mask]
            heat = heat.astype(np.uint8)
            img = np.copy(img)
            img = img.astype(np.uint8)
            img = img[laid:psize+laid, laid:psize+laid, :]
            pil_heat = Image.fromarray(heat)
            pil_img = Image.fromarray(img)
            pil_heat = pil_heat.resize(\
                    (psize//downby, psize//downby), Image.BICUBIC)
            pil_img = pil_img.resize(\
                    (psize//downby, psize//downby), Image.BICUBIC)
            np_heat = np.array(pil_heat).astype(np.uint8)
            np_img = np.array(pil_img).astype(np.uint8)

            #pil_mask_pred = Image.fromarray(mask_pred[i]*255)
            pil_mask_pred = Image.fromarray(pred[i]*255)
            pil_mask_pred = pil_mask_pred.resize(\
                    (psize//downby, psize//downby), Image.BICUBIC)
            np_mask_pred = np.array(pil_mask_pred)#/255).astype(np.uint8)
            Mask[left//scale//downby:left//scale//downby+psize//downby,\
                    top//scale//downby:top//scale//downby+psize//downby]\
                = np_mask_pred
            Heat[left//scale//downby:left//scale//downby+psize//downby,\
                    top//scale//downby:top//scale//downby+psize//downby,:]\
                = np_heat
            Ori[left//scale//downby:left//scale//downby+psize//downby,\
                    top//scale//downby:top//scale//downby+psize//downby,:]\
                = np_img 
            I = I +1

    Heat = Heat[:h//scale//downby, :w//scale//downby]
    Mask = Mask[:h//scale//downby, :w//scale//downby]
    Ori = Ori[:h//scale//downby, :w//scale//downby]

    return Ori, Mask, Heat

def downsize_slide(src_path, save_path, scale):
    #thumbnail is not accurate enough
    try:
        ts = large_image.getTileSource(src_path)
    except Exception as e:
        print(e)
        return
    save_dir = os.path.dirname(save_path)

    psize=256
    dataset = WSIDataset(ts, psize, scale)
    h = dataset.h
    w = dataset.w
    loader = DataLoader(dataset, batch_size=1,\
                shuffle=False, num_workers=20, pin_memory=True)
    
    Ori = np.zeros(\
            (h//scale+psize,w//scale+psize, 3)\
        ).astype(np.uint8)

    I = 0
    for batch in loader:
        ori = batch['ori']
        ori = ori.squeeze(0)
        ori = np.copy(ori)
        ori = ori.astype(np.uint8)

        jj = I//math.ceil(h/psize/scale)
        ii = I%math.ceil(h/psize/scale)
        top = jj * psize * scale
        left = ii * psize * scale
        pil_img = Image.fromarray(ori)
        pil_img = pil_img.resize((psize, psize), Image.BICUBIC)
        np_img = np.array(pil_img).astype(np.uint8)

        Ori[left//scale:left//scale+psize,\
                top//scale:top//scale+psize,:]\
            = np_img 
        I = I +1

    Ori = Ori[:h//scale, :w//scale]
    os.makedirs(save_dir, exist_ok = True)
    Image.fromarray(Ori).save(save_path)

def get_thumbnail(src_path, args):
    name = src_path.split('/')[-1].replace(f'.{args.src_ext}', '')
    src_dir = os.path.dirname(src_path)
    tar_dir = f'{src_dir}/thumbnail'
    tar_path = f'{tar_dir}/{name}.jpg'
    if os.path.exists(tar_path) == False:
        try:
            ts = large_image.getTileSource(src_path)
        except Exception as e:
            print(e)
            return
        os.makedirs(tar_dir, exist_ok=True)
        with open(tar_path, 'wb') as f:
            f.write(ts.getThumbnail()[0])
    return tar_path

def get_small_src(src_path, args):
    name = src_path.split('/')[-1].replace(f'.{args.src_ext}', '')
    tar_path = f'{args.save_src_dir}/small_src/{name}.{args.tar_ext}'
    return tar_path
    if os.path.exists(tar_path) == False:
    #if True: # small src quick
        scale = args.src_mag//args.net_mag*args.tar_scale
        downsize_slide(src_path, tar_path, scale)
    return tar_path
        

def task(src_path, net, device, args):
    name = src_path.split('/')[-1].replace(f'.{args.src_ext}', '')
    print(name)
    if check_result(src_path, args):
        #anno_path = f'{args.save_src_dir}/{args.sub_phase}/{name}_anno.{args.tar_ext}'
        anno_path = f'{args.save_result_dir}/{name}_anno.{args.tar_ext}'
        Mask = cv2.imread(\
                f'{args.save_result_dir}/{name}_pred.{args.tar_ext}',0)

        Anno = cv2.imread(anno_path, 0)
        #ori_path = f'{args.save_src_dir}/{args.sub_phase}/{name}.{args.tar_ext}' 
        #Ori = cv2.imread(ori_path)
    else:
        #print(name)
        anno_path =  f'{args.anno_dir}/{name}{args.anno_suffix}' 
        if os.path.exists(anno_path) == False:
            return
        small_src_path = get_small_src(src_path, args)
        if args.excluded_dir != None:
            box, Anno, excluded_mask = return_box_with_excluded_mask(small_src_path, anno_path, args) 
        else:
            box, Anno = return_box(small_src_path, anno_path, args)

        try:
            ts = large_image.getTileSource(src_path)
        except Exception as e:
            print(e)
            return

        dataset = SeWSIDataset(ts, box, args)
        Ori, Mask, Heat = test_wsi_dataset(dataset, args) #todo
        #resize
        Anno = cv2.resize(Anno, (Mask.shape[1],Mask.shape[0]))
        assert Anno.shape == Mask.shape,Anno.shape
        bg = np.any(Ori<=[225,225,225], axis=-1)
        bg = bg.astype(np.uint8)
        bg = cv2.resize(bg, (Mask.shape[1], Mask.shape[0]))
        bg = bg.astype(np.uint8)
        Anno = Anno/255 if np.max(Anno) > 1 else Anno
        Anno = Anno.astype(np.uint8)
        #if args.save_src_dir != 'sfva':
        Anno = Anno&bg

        if args.excluded_dir != None:
            print(name)
            excluded_mask = cv2.resize(excluded_mask, (Mask.shape[1],Mask.shape[0]))
            excluded_mask = excluded_mask.astype(np.bool)
            Mask[excluded_mask]=0

        #if args.save_src_dir:
        if args.save_result_dir:
            #tmp_dir = f'{args.save_src_dir}/{args.sub_phase}'
            tmp_dir = f'{args.save_result_dir}'
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_name = f'{tmp_dir}/{name}.{args.tar_ext}'
            if os.path.exists(tmp_name) == False:
                #print(Ori.shape)
                #Image.fromarray(Ori).save(tmp_name)
                #png.from_array(Ori, mode='RGB').save(tmp_name)
                assert True
        if args.save_result_dir:
            os.makedirs(args.save_result_dir, exist_ok=True)
            #tmp_name = f'{args.save_result_dir}/{name}_pred_laid{args.laid}.{args.tar_ext}'
            tmp_name = f'{args.save_result_dir}/{name}_pred.{args.tar_ext}'
            #png.from_array(Mask*255, mode='L').save(tmp_name)
            png.from_array(Mask, mode='L').save(tmp_name)
            #tmp_name = f'{args.save_result_dir}/{name}_heat_laid{args.laid}.{args.tar_ext}'
            tmp_name = f'{args.save_result_dir}/{name}_heat.{args.tar_ext}'
            #Image.fromarray(Heat).save(tmp_name)

        #if args.save_src_dir:
        if args.save_result_dir:
            #tmp_dir = f'{args.save_src_dir}/{args.sub_phase}'
            tmp_dir = f'{args.save_result_dir}'
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_name = f'{tmp_dir}/{name}_anno.{args.tar_ext}'
            bg = np.any(Ori<=[225,225,225], axis=-1)
            bg = bg.astype(np.uint8)
            #bg = cv2.resize(bg, (Mask.shape[1], Mask.shape[0]))
            bg = bg.astype(np.uint8)
            #Anno = cv2.resize(Anno, (Mask.shape[1],Mask.shape[0]))
            Anno = Anno/255 if np.max(Anno) > 1 else Anno
            Anno = Anno.astype(np.uint8)
            Anno = Anno&bg
            if os.path.exists(tmp_name) == False:
                #Image.fromarray(Anno*255).save(tmp_name)
                #png.from_array(Anno*255, mode='L').save(tmp_name)
                assert True
#    anno = cv2.resize(Anno, (Anno.shape[1]//10, Anno.shape[0]//10))
#    mask = cv2.resize(Mask, (Mask.shape[1]//10, Mask.shape[0]//10))
#    cmatrix, evals = evalmask(Anno, Mask)
#    v = evals
#    print('%s\tacc: %.2f\ttnr: %.2f\ttpr: %.2f\tppv:%.2f\tdice: %.2f'\
#            %(name.ljust(20), v['acc'], v['tnr'], v['tpr'],v['ppv'], v['dice']))
#
#    return cmatrix, evals
#
def multi_task(src_dir, net, device, args):
    src_paths = glob(f'{args.src}/*.{args.src_ext}')
    #src_paths = src_paths[:1]
    C = np.zeros((2,2))
    D=0
    count = 0
    for src_path in src_paths:
        name = src_path.split('/')[-1].replace(f'.{args.src_ext}', '')
        anno_path =  f'{args.anno_dir}/{name}{args.anno_suffix}' 
        if os.path.exists(anno_path) == False:
            continue
        #cmatrix, evals = task(src_path, net, device, args)
        task(src_path, net, device, args)
#        C = C + cmatrix
#        D = D + evals['dice']
#        count += 1
#    D = D/count
#    ACC = (C/C.sum()).trace()
#    TNR = C[0,0]/(C[0,0]+C[0,1])
#    TPR = C[1,1]/(C[1,1]+C[1,0])
#    PPV = C[1,1]/(C[1,1]+C[0,1])
#
#    #print('After All')
#    Cohort={'SFVA':'D3', 'Vanderbilt':'D4', 'UCSF':'D5'} 
#    cohort = args.src.split('/')[-1]
#    cohort = Cohort[cohort]
#    print(cohort, args.model_path)
#    print('PA: %.2f\tR: %.2f\tPPV: %.2f\tD: %.2f'%(ACC*100,TPR*100,PPV*100,D*100))
#



def get_args(flag=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path')
    parser.add_argument('--src')
    parser.add_argument('--anno-dir')
    parser.add_argument('--excluded-dir')
    parser.add_argument('--anno-suffix', default='_anno.png')
    parser.add_argument('--net-mag', type=int)
    parser.add_argument('--patch-size', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cpus', type=int, default=40)
    parser.add_argument('--src-mag', type=int, default=40)
    parser.add_argument('--save-src-dir')
    parser.add_argument('--sub-phase', default='sub_slide')
    parser.add_argument('--save-result-dir')
    parser.add_argument('--src-ext', default='tif')
    parser.add_argument('--tar-ext', default='png')
    parser.add_argument('--tar-scale', type=int, default=4)
    parser.add_argument('--anno-scale', type=int, default=16)
    parser.add_argument('--box-mode')
    parser.add_argument('--heAug', action='store_true')
    parser.add_argument('--laid', type=int, default=0)

    if flag:
        return parser.parse_args([])
    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    args = get_args()
    if args.src == None or args.model_path == None:
        print(args)
    elif os.path.isfile(args.src):
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        net = torch.load(args.model_path, map_location=device)
        net.eval()
        task(args.src, net, device, args)

    elif os.path.isdir(args.src):
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        net = torch.load(args.model_path, map_location=device)
        multi_task(args.src, net, device, args)
    end = time.time()
    #print('time:', (end-start)/60)
