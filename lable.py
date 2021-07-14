import math
import itertools
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET

class Rectangle:
    def __init__(self, points: ET.Element, canvas: (int,int)):
        self.ow = canvas[0]
        self.oh = canvas[1]
        self.category = points[0].text.strip()
        self.xmin = self.x_prop(float(points[4][0].text))
        self.ymin = self.y_prop(float(points[4][1].text))
        self.xmax = self.x_prop(float(points[4][2].text))
        self.ymax = self.y_prop(float(points[4][3].text))
        self.cenx = (self.xmax+self.xmin)/2
        self.ceny = (self.ymax+self.ymin)/2
        
        self.blocks = 1/7
        self.secarea = self.blocks*self.blocks

    def x_prop(self, x):
        return x/self.ow

    def y_prop(self, x):
        return x/self.oh

    def edges(self,new: (int,int)) -> ((int,int),(int,int)):
        procx = lambda x: int(x*new[1])
        procy = lambda x: int(x*new[0])
    
        top = (procx(self.xmin),procy(self.ymin))
        bottom = (procx(self.xmax),procy(self.ymax))
        return (top,bottom)

    def center(self) -> (int,int):
        
        cx = int(self.cenx/self.blocks)
        cy = int(self.ceny/self.blocks)

        return (cx,cy)

    def margins(self) -> (float,float):
        cell = self.center()
        margx = (self.cenx-cell[0]*self.blocks)/self.blocks
        margy = (self.ceny-cell[1]*self.blocks)/self.blocks

        return (margx,margy)
    
    def size(self) -> (float,float):
        width = self.xmax-self.xmin
        heigth = self.ymax-self.ymin
        return (width,heigth)
        
class Lable:
    def __init__(self, path: str, cats):
        tree = ET.parse(path)
        root =  tree.getroot()
        self.width = float(root[4][0].text)
        self.height = float(root[4][1].text)
        self.rectangles = list()
        self.prediction = torch.zeros([25,7,7], dtype=torch.float64)
        self.cats = cats

        for el in root.iter('object'):
            self.rectangles.append(Rectangle(el,(self.width,self.height)))

    def target(self) -> torch.Tensor:
        cells = {}
        for rectangle in self.rectangles:
            pos = rectangle.center()
            if(pos not in cells):
                cells[pos] = True
                self.prediction[4,pos[1],pos[0]] = 1
                cat = self.cats[rectangle.category]+5
                self.prediction[cat,pos[1],pos[0]] = 1

                margins = rectangle.margins()
                size = rectangle.size()

                self.prediction[0,pos[1],pos[0]] = margins[0]
                self.prediction[1,pos[1],pos[0]] = margins[1]
                self.prediction[2,pos[1],pos[0]] = size[0]
                self.prediction[3,pos[1],pos[0]] = size[1]

        return self.prediction

    def draw(self,image: np.ndarray) -> np.ndarray:
        for rectangle in self.rectangles:
            edges = rectangle.edges(image.shape)
            cv2.rectangle(image, edges[0], edges[1], (255,0,0), 2)
        return image

