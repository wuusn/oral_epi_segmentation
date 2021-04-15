""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, threshold=.15):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                    #返回1*1的池化结果
        self.threshold=threshold
        self.fc = nn.Sequential( 
            nn.Linear(channel, reduction, bias=False),  #W1=C/r*C （1,1,C/r)
            nn.ReLU(inplace=True),                                                     
            nn.Linear(reduction, channel, bias=False),  #W2=C*C/r （1,1,C）
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #view 相同的数据 不一样的大小（size）
        y = self.fc(y).view(b, c, 1, 1)
        y = y.expand_as(x)   #expend_as到和x一样的维度
        o = x * y
        if False:
        #if self.training == False:
            if torch.sum(o)/(o.shape[2]*o.shape[3])<= self.threshold:
                o=o*0
        return o

class TestNet(nn.Module):
    def __init__(self, n_channels, n_classes, threshold=.15):
        super(TestNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = False
        self.bn = True
        bilinear = False
        bn = True

        base = 64
        self.inc = DoubleConv(n_channels, base, bn)
        self.down1 = Down(base, base*2, bn)
        self.down2 = Down(base*2, base*4, bn)
        self.down3 = Down(base*4, base*8, bn)
        self.down4 = Down(base*8, base*8, bn)
        self.up1 = Up(base*16, base*4, bilinear, bn)
        self.up2 = Up(base*8, base*2, bilinear, bn)
        self.up3 = Up(base*4, base, bilinear, bn)
        self.up4 = Up(base*2, base, bilinear, bn)
        self.outc = OutConv(base, n_classes)
        self.se = SELayer(1, 1,threshold)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        o = self.outc(x)
        o = self.se(o)

        return o
