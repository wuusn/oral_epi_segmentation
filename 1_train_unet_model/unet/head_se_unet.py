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

class Head_SE_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bn=True):
        super(Head_SE_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.bn = bn

        self.inc = DoubleConv(n_channels, 64, bn)
        self.se = SELayer(64,16)
        self.down1 = Down(64, 128, bn)
        self.down2 = Down(128, 256, bn)
        self.down3 = Down(256, 512, bn)
        self.down4 = Down(512, 512, bn)
        self.up1 = Up(1024, 256, bilinear, bn)
        self.up2 = Up(512, 128, bilinear, bn)
        self.up3 = Up(256, 64, bilinear, bn)
        self.up4 = Up(128, 64, bilinear, bn)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.se(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
