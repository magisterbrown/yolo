import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from lable import Lable
import cv2
import random

class VocSet(Dataset):
    def __init__(self, annotations: str,img_path: str,boxes_path: str, categories: str, aug: float):
        self.lables = pd.read_csv(annotations,header = None,dtype=str)
        self.img_path = img_path
        self.boxes_path = boxes_path
        self.cats = {}
        self.aug = aug

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
        borh = int(oh*self.aug)
        ow = image.shape[1]
        borw = int(ow*self.aug)
        image=cv2.copyMakeBorder(image,borh,borh,borw,borw,cv2.BORDER_REPLICATE,value=[0,0,0])
        mina = 1-self.aug
        maxa = 1+self.aug
        
        scy = random.uniform(mina,maxa)
        scx = random.uniform(mina,maxa)
        shy= random.uniform(-self.aug,maxa-scy)
        shx= random.uniform(-self.aug,maxa-scx)
        lable.mov(shx,shy)
        y = int(oh*(shy+self.aug))
        x = int(ow*(shx+self.aug))

        
        lable.scale(scx,scy)
        heigth = int(oh*scy)
        width = int(ow*scx)
        image = image[y:y+heigth, x:x+width]
        #end of augmentations


        image =  cv2.resize(image, (448,448))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image - mean)/std
        image = image.transpose(2,0,1)

        return image, lable.target()
