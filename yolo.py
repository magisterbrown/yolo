import torch
import torch.nn as nn
from torchvision.models import resnet34

class Yolo(nn.Module):
    def __init__(self,resnet_weights: str):
        super().__init__()

        res = resnet34()
        weights = torch.load(resnet_weights)
        res.load_state_dict(weights)
        self.resnet34 = nn.Sequential(*list(res.children())[:-2])

        self.l1 = nn.Linear(25088,4096)
        self.rl = nn.ReLU()
        self.l2 = nn.Linear(4096,1470)
        self.pool = nn.MaxPool2d(2,2)
    def forward(self, x):

        x = self.resnet34(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.l1(x)
        x = self.rl(x)
        x = self.l2(x)
        x = torch.reshape(x,(-1,30,7,7))
        
        return x

