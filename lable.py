import math
import itertools
import torch
import xml.etree.ElementTree as ET

class Rectangle:
    def __init__(self, points: ET.Element, canvas: (int,int)):
        self.ow = canvas[0]
        self.oh = canvas[1]
        self.xmin = self.x_prop(float(points[4][0].text))
        self.ymin = self.y_prop(float(points[4][1].text))
        self.xmax = self.x_prop(float(points[4][2].text))
        self.ymax = self.y_prop(float(points[4][3].text))
        
        self.blocks = 1/7
        self.secarea = self.blocks*self.blocks

    # def regions(self) -> ((int,int),(int,int)):
    #     leftx = int(self.x_prop(self.xmin)/self.blocks)
    #     topy = int(self.y_prop(self.ymin)/self.blocks)
    #     rightx = int(self.x_prop(self.xmax)/self.blocks)
    #     boty = int(self.y_prop(self.ymin)/self.blocks)
    #     return ((leftx,rightx),(topy,boty))

    def x_prop(self, x):
        return x/self.ow

    def y_prop(self, x):
        return x/self.oh

    def edges(self,new: (int,int)) -> ((int,int),(int,int)):
        procx = lambda x: int(x*new[0])
        procy = lambda x: int(x*new[1])
    
        top = (procx(self.xmin),procy(self.ymin))
        bottom = (procx(self.xmax),procy(self.ymax))
        return (top,bottom)

    def center(self) -> (int,int):
        cx = (self.xmax+self.xmin)/2
        cy = (self.ymax+self.ymin)/2
        cx = int(cx/self.blocks)
        cy = int(cy/self.blocks)

        return (cx,cy)
    # def iou(self, square: (int, int)):
    #     xsecmin = square[0]*self.blocks
    #     xsecmax = (square[0]+1)*self.blocks
    #     ysecmin = square[1]*self.blocks
    #     ysecmax = (square[1]+1)*self.blocks
    #     width = max(0,min(xsecmax, self.x_prop(self.xmax)) - max(xsecmin, self.x_prop(self.xmin))) 
    #     height = max(0,min(ysecmax, self.y_prop(self.ymax)) - max(ysecmin, self.y_prop(self.ymin))) 
    #     intersection = width*height
    #     box = (self.x_prop(self.xmax-self.xmin) * self.y_prop(self.ymax - self.ymin)) 
    #     union = box+self.secarea-intersection

    #     return intersection/union
        
class Lable:
    def __init__(self, path: str):
        tree = ET.parse(path)
        root =  tree.getroot()
        self.root = root
        self.width = float(root[4][0].text)
        self.height = float(root[4][1].text)
        self.rectangles = list()
        self.prediction = torch.zeros([25,7,7], dtype=torch.float64)

        for el in root.iter('object'):
            self.rectangles.append(Rectangle(el,(self.width,self.height)))

    def get_prediction(self) -> torch.Tensor:
        for rectangle in self.rectangles:
            pass
        return self.prediction

    def add_rectangle(self, rectangle: Rectangle):
        print(rectangle.center())

    def target(self):
        pass
