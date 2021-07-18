import torch
from yolo import Yolo
import unittest
from torchvision.models import resnet34

class Test(unittest.TestCase):
    def setUp(self):
        self.yolo = Yolo("../checkpoints/resnet34-333f7ec4.pth")

    def layers(self):
        randim = torch.rand(1,3,448,448)
        pred = self.yolo(randim)
        print(pred.shape)