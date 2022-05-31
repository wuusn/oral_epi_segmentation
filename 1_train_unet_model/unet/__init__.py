from .unet_model import UNet
from .se_unet import *
from .tail_se_unet import Tail_SE_UNet
from .head_se_unet import Head_SE_UNet
from .test_net import TestNet
from .others import *
#from .yus_unet import *

def get_net_se():
    return SE_Unet(3,1,False,True)
def get_net_unet():
    return UNet(3,1,False,True)
def get_net_tail():
    return Tail_SE_UNet(3,1,False,True)
def get_net_head():
    return Head_SE_UNet(3,1,False,True)
def get_net_se4():
    return SE_Unet4(3,1,False,True)

