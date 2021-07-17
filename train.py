from torch.optim import optimizer
from loader import VocSet
from torch.utils.data import DataLoader
from yolo import Yolo
from loss import YoloLoss
from torch import optim
import math

epochs = 10
bach = 16


ds = VocSet("VOC2007/ImageSets/Layout/train.txt","VOC2007/JPEGImages","VOC2007/Annotations","VOC2007/categories.txt")
vs = VocSet("VOC2007/ImageSets/Layout/val.txt","VOC2007/JPEGImages","VOC2007/Annotations","VOC2007/categories.txt")
train_dataloader = DataLoader(ds, batch_size=bach, shuffle=True)
validation_dataloader = DataLoader(vs, batch_size=bach, shuffle=True)
criterion = YoloLoss(5,0.5)

yolo = Yolo("checkpoints/resnet34-333f7ec4.pth")
params = [{'params': yolo.resnet34b1.parameters(), 'lr': 0.01},
          {'params': yolo.resnet34b2.parameters(), 'lr': 0.01},
          {'params': yolo.linear.parameters(), 'lr':0.01}]

steps = int(math.ceil(len(train_dataloader)/bach))
optimizer = optim.Adam(params,weight_decay=0.0005)
sheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=[0.7,9,89],epochs=epochs,steps_per_epoch=steps)

# for ep in range(epochs):
#     for i, data in enumerate(train_dataloader):
#         inputs, labels = data