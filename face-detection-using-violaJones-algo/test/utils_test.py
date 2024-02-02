import unittest
from src.haar_features import HaarLikeFeature as haar
from src.haar_features import featureType
from src.utils import *
import numpy as np

'''
unittest: The Python unit testing framework
unittest supports test automation, sharing of setup and shutdown code for tests, aggregation of tests into collections, and independence of the tests from the reporting framework.
to run the unittest, python -m unittest test.utils_test
'''


class UtilsTest(unittest.TestCase):

    def setUp(self):
        # method called to prepare the test fixture
        self.haar_list = list()
        self.classifiers = None
        pass

    def tearDown(self):
        # method called immediately after the test method has been called and the result recorded
        pass

    def test_write_load_json_file(self):
        
        # prepare toy samples
        self.haar_list.append(haar(featureType.TWO_VERTICAL, (0, 0), 19, 19, 100000, 1))
        self.haar_list.append(haar(featureType.THREE_VERTICAL, (0, 0), 19, 19, 100000, 1, 1.5))
        self.haar_list.append(haar(featureType.FOUR, (0, 0), 19, 19, 100000, 1, .05))

        # write toy samples to a json file
        write_json_file(self.haar_list)
        # load toy sample from a json file
        self.classifiers = load_json_file()

        for c in self.haar_list:
            print("type: %s\tfeature type: %s\tposition: %s\twidth: %d\theight: %d\tthreshold: %d\tparity: %d\tweight: %f" % (
                type(c), c.type.name, str(c.top_left), c.width, c.height, c.threshold, c.parity, c.weight))

        for c in self.classifiers:
            print("type: %s\tfeature type: %s\tposition: %s\twidth: %d\theight: %d\tthreshold: %d\tparity: %d\tweight: %f" % (
                type(c), c.type, str(c.top_left), c.width, c.height, c.threshold, c.parity, c.weight))
        
        assert len(self.haar_list) == len(self.classifiers)
        for i in range(len(self.haar_list)):
            assert two_haar_equal(self.haar_list[i], self.classifiers[i])

if __name__ == "__main__":
    unittest.main()
