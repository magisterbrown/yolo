import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from lable import Lable
import cv2
import random

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
        name = self.lables.at[idx,0]
        lable = Lable(f'{self.boxes_path}/{name}.xml',self.cats)
        image = plt.imread(f'{self.img_path}/{name}.jpg')[...,::-1]/255


        #augmentations
        oh = image.shape[0]
        borh = int(oh*0.2)
        ow = image.shape[1]
        borw = int(ow*0.2)
        image=cv2.copyMakeBorder(image,borh,borh,borw,borw,cv2.BORDER_REPLICATE,value=[0,0,0])
        
        scy = random.uniform(0.8,1.2)
        scx = random.uniform(0.8,1.2)
        shy= random.uniform(-0.2,1.2-scy)
        shx= random.uniform(-0.2,1.2-scx)
        self.lable.mov(shx,shy)
        y = int(oh*(shy+0.2))
        x = int(ow*(shx+0.2))

        
        self.lable.scale(scx,scy)
        heigth = int(oh*scy)
        width = int(ow*scx)
        image = image[y:y+heigth, x:x+width]
        #end of augmentations


        image =  cv2.resize(image, (448,448))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image - mean)/std
        image = image.transpose(1,2,0)

        return image, lable.target()
