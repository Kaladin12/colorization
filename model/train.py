from time import sleep
from model import CNN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import glob
import torch
import torch.nn as nn
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt
from functools import lru_cache
from vars import PATH, GROUND_TRUTH

batch_size = 50

@lru_cache
def get_targets():
    filenames = [img for img in glob.glob(f"{GROUND_TRUTH}*.jpg")]
    images = [cv.cvtColor(cv.imread(img), cv.COLOR_BGR2LAB) for img in filenames]
    images = torch.from_numpy(np.asarray([images[i:i+batch_size] for i in range(0, len(images), batch_size)]))
    images = images.float()
    images = torch.reshape(images, (100, 50, 3,400,400))
    #print(images[0][:,1:].size())
    return images


def train(dataloader, model, loss_a, loss_b, optimizer):
    size = len(dataloader.dataset)
    print(size)
    target = get_targets()

    model.train()
    for batch, (x,y) in enumerate(dataloader):
        x = x.to('cpu')
        preds, _ = model(x)
        #pred = torch.squeeze(pred, 1)
        tgt_a, tgt_b = target[batch][:,1], target[batch][:,2]
        pred_a, pred_b = preds[:, 0], preds[:,1]
        lossA = loss_a(pred_a, tgt_a)
        lossB = loss_b(pred_b, tgt_b)
        print(f"loss: {lossA} {lossB}" )
        optimizer.zero_grad()
        lossA.backward()
        lossB.backward()
        optimizer.step()
        if ((batch+1)%100==0):
            print(f"loss:{lossA.item()}, {batch*len(x)}")

model = CNN().to('cpu')

loss_a = nn.CrossEntropyLoss()
loss_b = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#from torchvision import transforms
tr = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
training_data = torchvision.datasets.ImageFolder(PATH, tr)
train_dataloader = DataLoader(training_data, batch_size=batch_size)

train(train_dataloader, model, loss_a, loss_b, optimizer)
