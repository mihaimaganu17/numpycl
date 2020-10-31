import unittest
import numpy as np
from numpycl import numpycl as ncl

SIZE = 1920 * 1080

class NumpyCLTestCase(unittest.TestCase):
  def setUp(self):
    # Define class
    self.ncl = ncl.NumpyCL()
    # Define buffers
    self.array1 = np.random.randint(256, size=SIZE, dtype=np.uint8)
    self.array2 = np.random.randint(256, size=SIZE, dtype=np.uint8)
    self.sum_res = self.array1 + self.array2
    self.sub_res = self.array1 - self.array2


  def test_sum(self):
    self.assertEqual(self.ncl.sum(self.array1, self.array2).all(), 
                        self.sum_res.all(), "Incorect sum")


  def test_diff(self):
    self.assertEqual(self.ncl.sub(self.array1, self.array2).all(), 
                        self.sub_res.all(), "Incorect sub")

    
def suite():
  suite = unittest.TestSuite()
  suite.addTest(NumpyCLTestCase('test_sum'))
  suite.addTest(NumpyCLTestCase('test_diff'))
  return suite


if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  runner.run(suite())
