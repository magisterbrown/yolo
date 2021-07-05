import torch
from torch.utils.data import Dataset
import pandas as pd

class VocSet(Dataset):
    def __init__(self, annotations: str,img_path: str,boxes_path: str):
        self.lables = pd.read_csv(annotations,header = None,dtype=str)
        self.img_path = img_path
        self.boxes_path = boxes_path

    def __len__(self) -> int:
        return len(self.lables)

    def __getitem__(self, idx: int):
        pass
    
