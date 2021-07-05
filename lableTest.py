import unittest
from lable import Lable

class TestLable(unittest.TestCase):
    
    name = '006104'
    lable = Lable(f'VOC2007/Annotations/{name}.xml')
    def test_creation(self):
        
        self.lable.add_rectangle(self.lable.rectangles[2])
    
    def fields(self):
        print(self.lable.root.find("object")[0].text)

if __name__ == '__main__':
    print("df")
    unittest.main()
