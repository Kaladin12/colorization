import torch.nn as nn

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
        self.out = nn.Conv2d(in_channels=313, out_channels=2, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x = self.layer_1(x)
        #print(x.size())
        x = self.layer_2(x)
        #print(x.size())
        x = self.layer_3(x)
        #print(x.size())
        x = self.layer_4(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        #print(x.size())
        x = self.layer_5(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        #print(x.size())
        x = self.layer_6(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        #print(x.size())
        x = self.layer_7(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        #print(x.size())
        x = self.layer_8(x)
        #x = nn.Upsample(scale_factor=4, mode='bilinear')(x)
        #print("8: ",x.size())
        logits = self.softmax(x)
        #print("logits",logits.size())
        probs = self.out(logits)
        #print(probs.size())
        #return probs, probs
        return self.up(probs), probs

'''tr = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(mode='LAB')])
target_data = torchvision.datasets.ImageFolder(GROUND_TRUTH, tr)
target_dataloader = DataLoader(target_data, batch_size = batch_size)'''


