import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from lable import Lable
import cv2

class VocSet(Dataset):
    def __init__(self, annotations: str,img_path: str,boxes_path: str, categories: str):
        self.lables = pd.read_csv(annotations,header = None,dtype=str)
        self.img_path = img_path
        self.boxes_path = boxes_path
        self.cats = {}

        with open(categories) as f:
            for key,val in enumerate(f):
                self.cats[val.strip().lower()] = key

    def __len__(self) -> int:
        return len(self.lables)

    def __getitem__(self, idx: int):
        name = self.lables[int]
        lable = Lable(f'{self.boxes_path}/{self.name}.xml',self.cats)
        image = plt.imread(f'{self.img_path}/{self.name}.jpg')[...,::-1]/255
        image =  cv2.resize(image, (448,448))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image - mean)/std

        return image, lable.target()
    
