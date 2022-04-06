import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layer_6 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, dilation=2, padding=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layer_7 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layer_8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=313, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(in_channels=313, out_channels=2, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        #self.up = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        logits = self.softmax(x)
        probs = self.out(logits)
        return probs, probs
        #return self.up(probs), probs

model = CNN().to('cpu')



loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#training_data
from torchvision import transforms
tr = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
training_data = torchvision.datasets.ImageFolder('F:/MEGA/CETYS/sechs/vision_artificial/archive/data/train_black/', tr)
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)

def train(dataloader, model, loss_f, optimizer):
    size = len(dataloader.dataset)
    print(size)
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        x, y = x.to('cpu'), y.to('cpu')
        stuff, pred = model(x)
        #pred = torch.squeeze(pred, 1)
        print(pred)
        loss = loss_f(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch%100==0):
            print(f"loss:{loss.item()}, {batch*len(x)}")

train(train_dataloader, model, loss, optimizer)

        
