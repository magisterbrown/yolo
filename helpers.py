import torch
import cv2
import numpy as np
def draw(image: torch.Tensor,lable: torch.Tensor,pos: (int,int),cats: dict):
  box = lable[:,pos[0],pos[1]]
  parts = 1/lable.shape[-1]
  side = image.shape[-1]
  cat = float(torch.argmax(box[5:]))
  vl = int(float(torch.max(box[5:]))*float(box[4])*100)
  
  centerx = (pos[1]+box[0])*parts*side
  centery = (pos[0]+box[1])*parts*side

  x1 = int(centerx-box[2]*side/2)
  x2 = int(centerx+box[2]*side/2)
  y1 = int(centery-box[3]*side/2)
  y2 = int(centery+box[3]*side/2)

  color = np.random.rand(3,)
  image = image.cpu().numpy().transpose(1,2,0)
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  denorm = image*std+mean
  denorm = cv2.rectangle(denorm.copy() ,(x1,y1),(x2,y2),color,3)
  denorm=cv2.putText(denorm,f'{cats[cat]} {vl}%', (x1,y1-5),cv2.FONT_HERSHEY_TRIPLEX, 1, color)

  return denorm