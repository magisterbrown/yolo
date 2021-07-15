import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self):
        pass
    def forward(self,input: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
        pass