import unittest

from numpy.lib.function_base import sort_complex
from lable import Lable
import matplotlib.pyplot as plt
import cv2
import random

class TestLable(unittest.TestCase):
    
    name = '003301'
    def setUp(self):
        with open('../VOC2007/categories.txt') as f:
            cats = {}
            for key,val in enumerate(f):
                cats[val.strip().lower()] = key
        self.lable = Lable(f'../VOC2007/Annotations/{self.name}.xml',cats)

    
    def fields(self):
        print(self.lable.rectangles[1].category)
    
    def target(self):
        print(self.lable.target()[:,5,3])

    def draw(self):
        image = plt.imread(f'../VOC2007/JPEGImages/{self.name}.jpg')[...,::-1]/255
        #plt.imshow(image)
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
        #image = cv2.resize(image,None,fx=0.5, fy=2, interpolation =  cv2.INTER_LINEAR)
        image = self.lable.draw(image)
        plt.imshow(image)
        plt.show()
    
    
if __name__ == '__main__':
    unittest.main()
