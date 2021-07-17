import unittest
import sys
from loader import VocSet
from torch.utils.data import DataLoader
from loss import YoloLoss

class TestLoader(unittest.TestCase):

    def setUp(self):
        self.ds = VocSet("../VOC2007/ImageSets/Layout/train.txt","../VOC2007/JPEGImages","../VOC2007/Annotations","../VOC2007/categories.txt")
        self.train_dataloader = DataLoader(self.ds, batch_size=16, shuffle=True)

    def loadone(self):
        print(self.ds[0][1].shape)

    def loader(self):
        loss = YoloLoss(5,0.5)
        train_features, train_labels = next(iter(self.train_dataloader))
        loss(train_labels,train_labels)