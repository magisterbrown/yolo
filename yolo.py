import torch
import torch.nn as nn
from torchvision.models import resnet34

class Yolo(nn.Module):
    def __init__(self,resnet_weights: str):
        super().__init__()

        res = resnet34()
        weights = torch.load(resnet_weights)
        res.load_state_dict(weights)
        self.resnet34b1 = nn.Sequential(*list(res.children())[:-4])
        self.resnet34b2 = nn.Sequential(*list(res.children())[-4:-2])
        
        self.pool = nn.AvgPool2d(2,2)
        self.linear = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.1,inplace=True),
                nn.Linear(25088,4096),
                nn.LeakyReLU(negative_slope=0.1,inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096,1225)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x = self.resnet34b1(x)
        x = self.resnet34b2(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.linear(x)
        x = self.sig(x)*1.0001-0.00005
        x = torch.reshape(x,(-1,25,7,7))
        
        return x

