import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn

def get_vgg():
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 2)
    return model
