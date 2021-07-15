import unittest
from loader import VocSet
from torch.utils.data import DataLoader

class TestLoader(unittest.TestCase):

    def setUp(self):
        self.ds = VocSet("VOC2007/ImageSets/Layout/train.txt","VOC2007/JPEGImages","VOC2007/Annotations","VOC2007/categories.txt")
        self.train_dataloader = DataLoader(self.ds, batch_size=16, shuffle=True)
    def loadone(self):
        
        print(ds[0][1].shape)

    def loader(self):
        train_features, train_labels = next(iter(self.train_dataloader))