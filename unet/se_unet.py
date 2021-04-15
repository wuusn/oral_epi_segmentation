import torch.nn.functional as F
from .unet_parts import *

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential( 
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),                                                     
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bn=True):
        super(SE_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.bn = bn

        self.inc = DoubleConv(n_channels, 64, bn)
        self.se0 = SELayer(64,16)
        self.down1 = Down(64, 128, bn)
        self.se1 = SELayer(128,16)
        self.down2 = Down(128, 256, bn)
        self.se2 = SELayer(256,16)
        self.down3 = Down(256, 512, bn)
        self.se3 = SELayer(512,16)
        self.down4 = Down(512, 512, bn)
        self.se4 = SELayer(512,16)
        self.up1 = Up(1024, 256, bilinear, bn)
        self.up2 = Up(512, 128, bilinear, bn)
        self.up3 = Up(256, 64, bilinear, bn)
        self.up4 = Up(128, 64, bilinear, bn)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.se0(x1)
        x2 = self.down1(x1)
        x2 = self.se1(x2)
        x3 = self.down2(x2)
        x3 = self.se2(x3)
        x4 = self.down3(x3)
        x4 = self.se3(x4)
        x5 = self.down4(x4)
        x5 = self.se4(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SE_Unet4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bn=True):
        super(SE_Unet4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.bn = bn

        self.inc = DoubleConv(n_channels, 64, bn)
        self.se0 = SELayer(64,16)
        self.down1 = Down(64, 128, bn)
        self.se1 = SELayer(128,16)
        self.down2 = Down(128, 256, bn)
        self.se2 = SELayer(256,16)
        self.down3 = Down(256, 512, bn)
        self.se3 = SELayer(512,16)
        self.down4 = Down(512, 512, bn)
        #self.se4 = SELayer(512,16)
        self.up1 = Up(1024, 256, bilinear, bn)
        self.up2 = Up(512, 128, bilinear, bn)
        self.up3 = Up(256, 64, bilinear, bn)
        self.up4 = Up(128, 64, bilinear, bn)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.se0(x1)
        x2 = self.down1(x1)
        x2 = self.se1(x2)
        x3 = self.down2(x2)
        x3 = self.se2(x3)
        x4 = self.down3(x3)
        x4 = self.se3(x4)
        x5 = self.down4(x4)
        #x5 = self.se4(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

