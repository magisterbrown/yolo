import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
                    nn.Conv2d(3,64,7,2,padding=3,padding_mode='reflect'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU(),
                    nn.Conv2d(64,192,3,padding=1,padding_mode='reflect'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU(),
                    nn.Conv2d(192,128,1),
                    nn.ReLU(),
                    nn.Conv2d(128,256,3,padding=1,padding_mode='reflect'),
                    nn.ReLU(),
                    nn.Conv2d(256,256,1),
                    nn.ReLU(),
                    nn.Conv2d(256,512,3,padding=1,padding_mode='reflect'),
                    nn.MaxPool2d(2,2),
                    nn.ReLU()
                )
        repetable1 = [nn.Conv2d(512,256,1),nn.ReLU(),nn.Conv2d(256,512,3,padding=1,padding_mode='reflect'),nn.ReLU()]*4
        self.rp1 = nn.Sequential(*repetable1)
        self.l3 = nn.Sequential(
                nn.Conv2d(512,512,1),
                    nn.ReLU(),
                nn.Conv2d(512,1024,3,padding=1,padding_mode='reflect'),
                nn.MaxPool2d(2,2),
                    nn.ReLU()
                )
        repetable2 = [nn.Conv2d(1024,512,1),nn.ReLU(),nn.Conv2d(512,1024,3,padding=1,padding_mode='reflect'),nn.ReLU()]*2
        self.rp2 = nn.Sequential(*repetable2)
        self.l5 = nn.Sequential(
                nn.Conv2d(1024,1024,3,padding=1,padding_mode='reflect'),
                    nn.ReLU(),
                nn.Conv2d(1024,1024,3,stride=2,padding=1,padding_mode='reflect'),
                    nn.ReLU(),
                nn.Conv2d(1024,1024,3,padding=1,padding_mode='reflect'),
                    nn.ReLU(),
                nn.Conv2d(1024,1024,3,padding=1,padding_mode='reflect'),
                    nn.ReLU()
                )
        self.linear = nn.Sequential(
                nn.Linear(50176,4096),
                nn.ReLU(),
                nn.Linear(4096,1470)
                )


    def forward(self, x):
        x = self.l1(x)
        x = self.rp1(x)
        x = self.l3(x)
        x = self.rp2(x)
        x = self.l5(x)
        return x

