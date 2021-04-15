import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * #QApplication, QWidget
from unet import UNet
import torch
from glob import glob
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn
import png
import os
from matplotlib import cm
from utils.dataset import *
from torch.utils.data import DataLoader


class filedialogdemo(QWidget):
    def __init__(self, parent = None):
        super(filedialogdemo, self).__init__(parent)
                
        layout = QVBoxLayout()
        self.setFixedWidth(800)
        self.setFixedHeight(700)

        self.btn_model = QPushButton("select model path")
        self.btn_model.clicked.connect(self.getModel)
        layout.addWidget(self.btn_model)

        self.btn_src = QPushButton("select src WSI path")
        self.btn_src.clicked.connect(self.getSrc)
        layout.addWidget(self.btn_src)

        self.btn_run = QPushButton("run")
        self.btn_run.clicked.connect(self.runTest)
        layout.addWidget(self.btn_run)

        self.le = QLabel('')
        self.le.setScaledContents(True)
        layout.addWidget(self.le)

        self.setLayout(layout)
        self.setWindowTitle('test WSI')
		
    def getModel(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/mnt/md0/_datasets/OralCavity/tma@10_2/save/milesialx2_cb_again/checkpoints' )
        self.btn_model.setText(fname[0])
        self.btn_run.setText('run')
    def getSrc(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/mnt/md0/_datasets/OralCavity/WSI')
        self.btn_src.setText(fname[0])
        self.btn_run.setText('run')

    def show_tmp(self):
        self.le.setPixmap(QPixmap('/tmp/tmp.png'))

    def runTest(self):
        timer = QTimer()
        timer.timeout.connect(self.show_tmp)
        model_path = self.btn_model.text()
        src_path = self.btn_src.text()
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        net = torch.load(model_path, map_location=device)
        net.eval()
        im_path = src_path
        try:
            ts = large_image.getTileSource(im_path)
            w = ts.sizeX
            h = ts.sizeY
        except Exception as e:
            print(e)
            return
        psize=256
        batch_size=10
        scale = 4
        downby = 4
        dataset = WSIDataset(ts, psize, scale)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=20, pin_memory=True)
        Heat = np.zeros((h//scale//downby +psize,w//scale//downby +psize, 3)).astype(np.uint8)
        Ori = np.zeros((h//scale//downby +psize,w//scale//downby +psize, 3)).astype(np.uint8)
        Mask = np.zeros((h//scale//downby +psize,w//scale//downby +psize)).astype(np.uint8)
        I = 0
        count = 0
        timer.start(10000)
        for batch in loader:
            imgs = batch['image']
            imgs = imgs.to(device=device, dtype=torch.float32)
            output = net(imgs)
            output = output.squeeze(1)
            if net.n_classes > 1:
                output = torch.sigmoid(output)
                output = output.detach().cpu().numpy()
                pred = output[:,2,:,:]
                #output = np.moveaxis(output,0,-1)
                #mask_pred = np.argmax(output, axis=1)
                mask_pred = (pred>.5).astype(np.uint8) # b,h,w

            else:
                output = torch.sigmoid(output)
                pred = output.detach().cpu().numpy()
                mask_pred = (pred>.5).astype(np.uint8) # b,h,w
            
            nb = pred.shape[0]
            count += nb
            for i in range(nb):
                jj = I//math.ceil(h/psize/scale)
                ii = I%math.ceil(h/psize/scale)
                top = jj * psize * scale
                left = ii * psize * scale
                jet_pred = cm.jet(pred[i])[:,:,:3]*255
                jet_pred = jet_pred.astype(np.uint8)
                bool_mask = pred[i]>.5
                img = batch['ori'][i]#imgs[i].detach().cpu().numpy()
                #img = np.moveaxis(img, 0, -1)
                heat = np.copy(img)
                heat[bool_mask] = jet_pred[bool_mask]
                heat = heat.astype(np.uint8)
                img = np.copy(img)
                img = img.astype(np.uint8)
                #
                pil_heat = Image.fromarray(heat)
                pil_img = Image.fromarray(img)
                pil_heat = pil_heat.resize((psize//downby, psize//downby), Image.BICUBIC)
                pil_img = pil_img.resize((psize//downby, psize//downby), Image.BICUBIC)
                np_heat = np.array(pil_heat).astype(np.uint8)
                np_img = np.array(pil_img).astype(np.uint8)

                pil_mask_pred = Image.fromarray(mask_pred[i]*255)
                pil_mask_pred = pil_mask_pred.resize((psize//downby, psize//downby), Image.BICUBIC)
                np_mask_pred = (np.array(pil_mask_pred)/255).astype(np.uint8)
                #print(np_mask_pred.shape)
                #
                #print(h,w,left,top)
                Mask[left//scale//downby:left//scale//downby+psize//downby, top//scale//downby:top//scale//downby+psize//downby] = np_mask_pred
                Heat[left//scale//downby:left//scale//downby+psize//downby, top//scale//downby:top//scale//downby+psize//downby,:] = np_heat
                Ori[left//scale//downby:left//scale//downby+psize//downby, top//scale//downby:top//scale//downby+psize//downby,:] = np_img 
                I = I + 1
            if count % 500 ==0:
                Image.fromarray(Heat[:h//scale//downby, :w//scale//downby]).save('/tmp/tmp.png')
            #    return
           #self.le.setPixmap(QPixmap('/mnt/md0/_datasets/OralCavity/tma@10_2/save/milesialx2_cb_again/sfva_result/SP06-1112 D5_heat.png'))
           #self.le.setPixmap(QPixmap('/home/yxw1452/non_epi.png' ))
        Image.fromarray(Heat[:h//scale//downby, :w//scale//downby]).save('/tmp/tmp.png')
        self.le.setPixmap(QPixmap('/tmp/tmp.png'))
        self.btn_run.setText('done!')
		
				
def main():
   app = QApplication(sys.argv)
   ex = filedialogdemo()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
