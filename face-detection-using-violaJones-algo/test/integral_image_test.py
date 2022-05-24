import unittest
from src.integral_image import IntegralImage as integral
from src.integral_image import get_sum
import numpy as np
import math
import random
from PIL import Image

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.integral_image_test
'''


class IntegralImageTest(unittest.TestCase):

    def setUp(self):
        # method called to prepare the test fixture
        self.ori_img = np.array(Image.open('./train_images/FACES/face00001.bmp'), dtype=np.float64)
        self.int_img = integral(self.ori_img).int_img
        self.ori_img_sq = integral(self.ori_img).img_sq
        self.int_img_sq = integral(self.ori_img).int_img_sq

    def tearDown(self):
        # method called immediately after the test method has been called and the result recorded
        pass

    def test_square_image(self):
        random.seed(10)
        for _ in range(5):
            x, y = int(random.random()*19), int(random.random()*19)
            assert self.ori_img_sq[x][y] == math.pow(self.ori_img[x][y], 2)

    def test_integral_calculation(self):
        # top-left corner
        assert self.int_img[1, 1] == self.ori_img[0, 0] 
        # bottom-left corner
        assert self.int_img[-1, 1] == np.sum(self.ori_img[:, 0])
        # top-right corner
        assert self.int_img[1, -1] == np.sum(self.ori_img[0, :])
        # bottom-right corner
        assert self.int_img[-1, -1] == np.sum(self.ori_img)

    def test_integral_sq_calculation(self):
         # top-left corner
        assert self.int_img_sq[1, 1] == self.ori_img_sq[0, 0] 
        # bottom-left corner
        assert self.int_img_sq[-1, 1] == np.sum(self.ori_img_sq[:, 0])
        # top-right corner
        assert self.int_img_sq[1, -1] == np.sum(self.ori_img_sq[0, :])
        # bottom-right corner
        assert self.int_img_sq[-1, -1] == np.sum(self.ori_img_sq)
        
    def test_get_sum(self):
        # pay attention that integral image has additional rows or columns of 0
        assert get_sum(self.int_img, (0, 0), (1, 1)) == self.ori_img[0, 0]
        assert get_sum(self.int_img, (0, 0), (-1, -1)) == np.sum(self.ori_img)

    def test_variance(self):
        assert round(integral(self.ori_img).variance, 5) == round(np.var(self.ori_img), 5)


if __name__ == "__main__":
    # a command-line program that loads a set of tests from integral_image and runs them
    unittest.main()
