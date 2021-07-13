import unittest
from lable import Lable

class TestLable(unittest.TestCase):
    
    name = '006104'
    def setUp(self):
        with open('VOC2007/categories.txt') as f:
            cats = {}
            for key,val in enumerate(f):
                cats[val.strip()] = key
            print(f'{cats}')
        self.lable = Lable(f'VOC2007/Annotations/{self.name}.xml')
    def test_creation(self):
        self.lable.add_rectangle(self.lable.rectangles[2])
    
    def fields(self):
        print(self.lable.rectangles[1].category)
    
    def target(self):
        print(type(self.lable.target()))

    

if __name__ == '__main__':
    print("df")
    unittest.main()
