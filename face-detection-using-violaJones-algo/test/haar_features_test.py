import unittest
from src.haar_features import HaarLikeFeature as haar
from src.haar_features import featureType
from src.integral_image import IntegralImage as integral
from src.integral_image import get_sum
import numpy as np
from PIL import Image

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.haar_features_test
'''

size = (24, 24) # resize the image with even-number width & height


class HarrLikeFeatureTest(unittest.TestCase):

    def setUp(self):
        # method called to prepare the test fixture
        img_arr = np.array(Image.open('./train_images/FACES/face00001.bmp'), dtype=np.float64)
        img_arr.resize(size)
        self.int_img = integral(img_arr).int_img

    def tearDown(self):
        # method called immediately after the test method has been called and the result recorded
        pass

    def test_two_vertical(self):
        # check the Two-rectangle(vertical) features
        # included fail test in which parity == -1, indicating misclassification
        feature = haar(featureType.TWO_VERTICAL, (0, 0), 24, 24, 100000, 1)
        white = get_sum(self.int_img, (0, 0), (24, 12))
        grey = get_sum(self.int_img, (0, 12), (24, 24))
        expected = 1 if feature.threshold * feature.parity > white - grey else 0
        expected_fail = 1 if feature.threshold * -1 > white - grey else 0

        assert feature.get_vote(self.int_img, 1) == expected
        assert feature.get_vote(self.int_img, 1) != expected_fail

    def test_two_horizontal(self):
        # check the Two-rectangle(horizontal) features
        # included fail test in which parity == -1, indicating misclassification
        feature = haar(featureType.TWO_HORIZONTAL, (0, 0), 24, 24, 100000, 1)
        white = get_sum(self.int_img, (0, 0), (24, 12))
        grey = get_sum(self.int_img, (0, 12), (24, 24))
        expected = 1 if feature.threshold * feature.parity > white - grey else 0
        expected_fail = 1 if feature.threshold * -1 > white - grey else 0

        assert feature.get_vote(self.int_img) == expected
        assert feature.get_vote(self.int_img) != expected_fail

    def test_three_vertical(self):
        # check the Three-rectangle(vertical) features
        # included fail test in which parity == -1, indicating misclassification
        feature = haar(featureType.THREE_VERTICAL, (0, 0), 24, 24, 100000, 1)
        white = get_sum(self.int_img, (0, 0), (8, 24))
        grey = get_sum(self.int_img, (0, 8), (24, 16))
        white += get_sum(self.int_img, (0, 16), (24, 24))
        expected = 1 if feature.threshold * feature.parity > white - grey else 0
        expected_fail = 1 if feature.threshold * -1 > white - grey else 0

        assert feature.get_vote(self.int_img) == expected
        assert feature.get_vote(self.int_img) != expected_fail

    def test_three_horizontal(self):
        # check the Three-rectangle(horizontal) features
        # included fail test in which parity == -1, indicating misclassification
        feature = haar(featureType.THREE_HORIZONTAL, (0, 0), 24, 24, 100000, 1)
        white = get_sum(self.int_img, (0, 0), (24, 8))
        grey = get_sum(self.int_img, (8, 0), (24, 16))
        white += get_sum(self.int_img, (16, 0), (24, 24))
        expected = 1 if feature.threshold * feature.parity > white - grey else 0
        expected_fail = 1 if feature.threshold * -1 > white - grey else 0

        assert feature.get_vote(self.int_img) == expected
        assert feature.get_vote(self.int_img) != expected_fail

    def test_four(self):
        # check the Four-rectangle features
        # included fail test in which parity == -1, indicating misclassification
        feature = haar(featureType.FOUR, (0, 0), 24, 24, 100000, 1)
        white = get_sum(self.int_img, (0, 0), (12, 12))
        grey = get_sum(self.int_img, (12, 0), (24, 12))
        grey += get_sum(self.int_img, (0, 12), (12, 24))
        white += get_sum(self.int_img, (12, 12), (24, 24))
        expected = 1 if feature.threshold * feature.parity > white - grey else 0
        expected_fail = 1 if feature.threshold * -1 > white - grey else 0

        assert feature.get_vote(self.int_img) == expected
        assert feature.get_vote(self.int_img) != expected_fail


if __name__ == "__main__":
    unittest.main()
