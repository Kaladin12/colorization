import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from torchsummary import summary


training_data = datasets.ImageNet(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)