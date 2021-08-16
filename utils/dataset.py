import math
from torch.utils.data import Dataset
from glob import glob
from random import random
from random import uniform
from os.path import exists
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from scipy.ndimage.morphology import binary_dilation
import numpy as np
import torch
import large_image
#from histomicstk.preprocessing.augmentation.\
#                                            color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration

def get_gradient(np_im):
    r = np_im[:,:,0]
    row, column = r.shape
    gradient = np.zeros((row, column))
    rf = np.copy(r)
    rf = rf.astype('float')
    for x in range(row-1):
        for y in range(column -1):
            gx = abs(rf[x+1,y]-rf[x,y])
            gy = abs(rf[x,y+1]-rf[x,y])
            gradient[x,y] = gx+gy
    gradient = gradient.astype('uint8')
    return gradient

class SeWSIDataset(Dataset):
    def __init__(self, ts, box, args):
        self.ts = ts
        self.box = box
        self.scale = args.src_mag // args.net_mag
        self.psize = args.patch_size
        self.laid = args.laid
        self.Left = box[0]
        self.Top = box[1]
        self.Right = box[2]
        self.Bottom = box[3]
        self.size = self.scale * self.psize
        self.size_laid = self.scale * (self.psize+2*self.laid)
        self.w = self.Bottom - self.Top
        self.h = self.Right - self.Left
        self.length = math.ceil(self.h/self.size)*math.ceil(self.w/self.size)
        self.heAug = args.heAug
    def __len__(self):
        return self.length
    def __getitem__(self,i):
        jj = i//math.ceil(self.h/self.size)
        ii = i%math.ceil(self.h/self.size)
        left = jj * self.size + self.Top
        top = ii * self.size + self.Left
        left_laid = left-self.laid*self.scale
        top_laid = top-self.laid*self.scale
        patch, _ = self.ts.getRegion(
                region = dict(left=left_laid,top=top_laid, width=self.size_laid, height=self.size_laid),
                format = large_image.tilesource.TILE_FORMAT_PIL)
        patch = patch.convert(mode='RGB')
        patch = patch.resize((self.psize+2*self.laid, self.psize+2*self.laid), Image.BICUBIC)
        np_patch = np.array(patch).astype(np.uint8)
        np_img= np_patch.transpose((2,0,1))
        np_img = np_img/255
#        if self.heAug:
#            heFail = False
#            try:
#                he_patch1 = rgb_perturb_stain_concentration(patch)
#            except:
#                heFail = True
#                pass
#            if heFail is not True:
#                he_patch2 = rgb_perturb_stain_concentration(patch)
#                he_patch3 = rgb_perturb_stain_concentration(patch)
#            else:
#
#            np_he1 = he_patch1.transpose((2,0,1))
#            np_he1 = np_he1/255
#            he1 = torch.from_numpy(np_he1)
#            np_he2 = he_patch2.transpose((2,0,1))
#            np_he2 = np_he2/255
#            he2 = torch.from_numpy(np_he2)
#            np_he3 = he_patch3.transpose((2,0,1))
#            np_he3 = np_he3/255
#            he3 = torch.from_numpy(np_he3)
#            return {'image': torch.from_numpy(np_img), 'ori':np_patch, \
#                'he1':he1,  }

        return {'image': torch.from_numpy(np_img), 'ori':np_patch}
class WSIDataset(Dataset):
    def __init__(self, ts, psize=256, scale=4):
        self.ts = ts
        self.w = ts.sizeX
        self.h = ts.sizeY
        self.scale = scale
        self.psize = psize
        self.size = self.scale * self.psize
        self.length = math.ceil(self.h/self.size)*math.ceil(self.w/self.size)
    def __len__(self):
        return self.length
    def __getitem__(self,i):
        jj = i//math.ceil(self.h/self.size)
        ii = i%math.ceil(self.h/self.size)
        left = jj * self.size
        top = ii * self.size
        patch, _ = self.ts.getRegion(
                region = dict(left=left,top=top, width=self.size, height=self.size),
                format = large_image.tilesource.TILE_FORMAT_PIL)
        patch = patch.convert(mode='RGB')
        patch = patch.resize((self.psize, self.psize), Image.BICUBIC)
        np_patch = np.array(patch).astype(np.uint8)
        if('self.mode' == 'gradient'):
            np_img = get_gradient(np_patch)
            np_img = np.expand_dims(np_img, axis=0)
        else:
            np_img= np_patch.transpose((2,0,1))
        np_img = np_img/255
        return {'image': torch.from_numpy(np_img), 'ori':np_patch}

class TMADataset(Dataset):
    def __init__(self, dir, phase='train', args=False):
        self.dir = dir # /path/train/label/*.png
        self.phase = phase
        self.args = args
        self.im_paths = [f for f in glob(f'{dir}/*/*.png') if not f.endswith('_mask.png')]
        self.label2idx = {'epi':1,'nontumor':1, 'tumor':2}
    def __len__(self):
        return len(self.im_paths)
    def __getitem__(self, i):
        im_path = self.im_paths[i]
        mask_path = im_path.replace('.png', '_mask.png')
        label = im_path.split('/')[-2]
        label = self.label2idx[label]
        im = Image.open(im_path)
        np_im = np.array(im)
        mask = Image.open(mask_path)
        np_mask = np.array(mask).astype(np.uint8)
        if self.phase == 'train':
            if random() > .5:
                im = TF.hflip(im)
                mask = TF.hflip(mask)
            if random() > .5:
                im = TF.vflip(im)
                mask = TF.vflip(mask)
            np_mask = np.array(mask).astype(np.uint8)
            np_im = np.array(im)
            if self.args.he_color == True:
                try:
                    np_im = rgb_perturb_stain_concentration(np_im)
                except:
                    pass
        if(self.args.gradient==True):
            np_im = get_gradient(np_im)
            np_im = np.expand_dims(np_im, 0)
        else:
            np_im = np_im.transpose((2,0,1))
        np_im = np_im/255
        np_mask = np_mask/255 if np_mask.max() > 1 else np_mask
        #np_mask = (np_mask*label).astype(np.uint8)
        np_mask = (np_mask).astype(np.uint8)
        return {'image': torch.from_numpy(np_im), 'mask':torch.from_numpy(np_mask), 'edge_weight': False}


class BasicDataset(Dataset):
    def __init__(self, dir, phase='train', args=False):
        # dir/group/id/name.png, dir/group/id/name_mask.png
        self.dir = dir
        self.phase = phase
        self.args = args
        self.img_paths = [f for f in glob(f'{dir}/*/*/*.png') if not f.endswith('_mask.png')]
        #self.img_paths = [f for f in glob(f'{dir}/*/*.png') if not f.endswith('_mask.png')]
        self.label2idx = {'epi':1, 'nontumor':0, 'tumor':1}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        mask_path = img_path.replace('.png', '_mask.png')
        pil_img = Image.open(img_path)
        if self.args.tumor:
            label = img_path.split('/')[-3]
            label = self.label2idx[label]
        if exists(mask_path):
            pil_mask = Image.open(mask_path).convert(mode='1')

            if self.phase == 'train':
                if random() > .5:
                    pil_img = TF.hflip(pil_img)
                    pil_mask = TF.hflip(pil_mask)
                if random() > .5:
                    pil_img = TF.vflip(pil_img)
                    pil_mask = TF.vflip(pil_mask)
                if random() > .5:
                    pil_img = TF.adjust_brightness(pil_img, uniform(1, 1.4))
                if random() > .5:
                    pil_img = TF.adjust_contrast(pil_img, uniform(1, 1.4))
                if random() > .5:
                    pil_img = TF.adjust_saturation(pil_img, uniform(1, 1.4))
                if random() > .5:
                    pil_img = TF.adjust_hue(pil_img, uniform(-.5, .5))
                if random() > .5:
                    pil_img = pil_img.filter(ImageFilter.GaussianBlur(int(random()>.5)+1))


            np_mask = np.array(pil_mask).astype(np.uint8)
            edge_weight = binary_dilation(np_mask==1, iterations=2) & ~np_mask #
            edge_weight = edge_weight.astype(np.uint8)
#
#            tensor_img = ToTensor()(pil_img)
#            tensor_mask = ToTensor()(pil_mask)
#            tensor_edge_weight = ToTensor()(edge_weight)
#
#            return {'image': tensor_img, 'mask': tensor_mask, 'edge_weight': tensor_edge_weight}
#
#        return {'image': ToTensor()(pil_img)}

            np_img = np.array(pil_img)
            if self.args.he_color == True and self.phase == 'train':
                try:
                    np_img = rgb_perturb_stain_concentration(np_img)
                except:
                    pass
            if(self.args.gradient==True):
                np_img = get_gradient(np_img)
                np_img = np_img.reshape(1,256,256)
            else:
                np_img = np_img.transpose((2,0,1))
            np_img = np_img/255
            np_mask = np_mask/255 if np_mask.max() > 1 else np_mask
            if self.args.tumor:
                np_mask = np_mask*label
            return {'image': torch.from_numpy(np_img), 'mask': torch.from_numpy(np_mask), 'edge_weight': torch.from_numpy(edge_weight)}
        if self.args.gradient==True:
            np_img = get_gradient(np_img)
        return {'image': torch.from_numpy(np_img/255)}




class ClassDataset(Dataset):
    def __init__(self, dir, phase='train', args=False):
        self.dir = dir # /path/train/label/*.png
        self.phase = phase
        self.args = args
        self.mask_paths = glob(f'{dir}/nontumor/*_mask.png')
        self.mask_paths.extend(glob(f'{dir}/tumor/*_mask.png'))
        self.label2idx = {'nontumor':0, 'tumor':1}
    def __len__(self):
        return len(self.mask_paths)
    def __getitem__(self, i):
        mask_path = self.mask_paths[i]
        im_path = mask_path.replace('_mask.png', '.png')
        #mask_path = im_path.replace('.png', '_mask.png')
        label = im_path.split('/')[-2]
        label = self.label2idx[label]
        im = Image.open(im_path)
        np_im = np.array(im)
        #mask = Image.open(mask_path)
        #np_mask = np.array(mask).astype(np.uint8)
        if self.phase == 'train':
            if random() > .5:
                im = TF.hflip(im)
                #mask = TF.hflip(mask)
            if random() > .5:
                im = TF.vflip(im)
                #mask = TF.vflip(mask)
            #np_mask = np.array(mask).astype(np.uint8)
            np_im = np.array(im)
            if self.args.he_color == True:
                try:
                    np_im = rgb_perturb_stain_concentration(np_im)
                except:
                    pass
        if(self.args.gradient==True):
            np_im = get_gradient(np_im)
            np_im = np.expand_dims(np_im, 0)
        else:
            np_im = np_im.transpose((2,0,1))
        np_im = np_im/255
        #np_mask = np_mask/255 if np_mask.max() > 1 else np_mask
        #np_mask = (np_mask*label).astype(np.uint8)
        #np_mask = (np_mask).astype(np.uint8)
        return {'image': torch.from_numpy(np_im), 'label':label, 'edge_weight': False}








