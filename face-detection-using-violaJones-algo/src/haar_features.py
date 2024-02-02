from src.integral_image import IntegralImage as integral
from src.integral_image import get_sum
from enum import Enum


class featureType(Enum):
    # featureType .value to access the tuple
    # tuple(width, height)
    TWO_VERTICAL = (1, 2)
    TWO_HORIZONTAL = (2, 1)
    THREE_VERTICAL = (1, 3)
    THREE_HORIZONTAL = (3, 1)
    FOUR = (2, 2)


class HaarLikeFeature(object):
    # class for getting harr-like features
    # how to select the weak learner (the single rectangle feature selection)
    # h_j(x) = 1 if p_j*f_j(x) < p_j*threshold
    # h_j(x) = 0 otherwise

    def __init__(self, feature_type, position, width, height, threshold, parity, weight=1):
        '''
        : para feature_type: 5 enumerate types in total
        : para position: position of sub-window (top-left)
        : para width, height: size of the image
        : para threshold: min number of misclassified examples
        : para parity: indicating the direction of the inequality (+1 or -1)
        '''
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0]+width, position[1]+height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.parity = parity
        self.weight = weight

    def calc_score(self, int_img):
        score, white, grey = 0, 0, 0

        if self.type == featureType.TWO_VERTICAL:
            white += get_sum(int_img, self.top_left, 
                (int(self.top_left[0] + self.width), int(self.top_left[1] + self.height / 2)))
            grey += get_sum(int_img, (self.top_left[0], 
                int(self.top_left[1] + self.height / 2)), self.bottom_right)

        elif self.type == featureType.TWO_HORIZONTAL:
            white += get_sum(int_img, self.top_left,
                (int(self.top_left[0] + self.width/2), self.top_left[1] + self.height))
            grey += get_sum(int_img,
                (int(self.top_left[0] + self.width/2), self.top_left[1]), self.bottom_right)
            
        elif self.type == featureType.THREE_VERTICAL:
            white += get_sum(int_img, self.top_left,
                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            grey += get_sum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)), 
                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            white += get_sum(int_img, (self.top_left[0],
                int(self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

        elif self.type == featureType.THREE_HORIZONTAL:
            white += get_sum(int_img, self.top_left,
                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            grey += get_sum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            white += get_sum(int_img, (self.top_left[0], int(
                self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

        elif self.type == featureType.FOUR:
            white += get_sum(int_img, self.top_left,
                (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            grey += get_sum(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            grey += get_sum(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            white += get_sum(int_img, (int(self.top_left[0] + self.width / 2),
                int(self.top_left[1] + self.height / 2)), self.bottom_right)
            
        score = white - grey
        return score

    def get_vote(self, int_img):
        # Get the vote of this feature for a given integral image (prediction)
        # note h_j(x) = 0 otherwise
        score = self.calc_score(int_img)
        return self.weight * (1 if score < self.parity * self.threshold else 0)
