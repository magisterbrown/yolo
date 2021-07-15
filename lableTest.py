import unittest
from lable import Lable
import matplotlib.pyplot as plt

class TestLable(unittest.TestCase):
    
    name = '007526'
    def setUp(self):
        with open('VOC2007/categories.txt') as f:
            cats = {}
            for key,val in enumerate(f):
                cats[val.strip().lower()] = key
        self.lable = Lable(f'VOC2007/Annotations/{self.name}.xml',cats)

    
    def fields(self):
        print(self.lable.rectangles[1].category)
    
    def target(self):
        print(self.lable.target()[:,5,3])

    def draw(self):
        image = plt.imread(f'VOC2007/JPEGImages/{self.name}.jpg')[...,::-1]/255
        image = self.lable.draw(image)
        plt.imshow(image)
        plt.show()
    
    
if __name__ == '__main__':
    unittest.main()
