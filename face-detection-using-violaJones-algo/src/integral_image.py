import numpy as np
import math

# A summed-area table, as known as an integral image
# It is produced by cumulative addition of intensities on subsequent pixels in horizontal and vertical axis
# It also greatly reduce the computation of features


class IntegralImage(object):
    # Class for getting integral image
    # attribute int_img is the calculated integral image of a bigger size than original

    def __init__(self, img):
        self.shape = (img.shape[0] + 1, img.shape[1] + 1)
        self.img = img
        self.img_sq = img * img
        # integral image to be calculated
        self.int_img = np.ones(self.shape)
        # integral image squard to be calculated (for nomalization)
        self.int_img_sq = np.ones(self.shape)
        # memo that indicates if this position already calc
        self.memo = np.zeros(self.shape)
        self.variance = 0.0
        self.get()
        self.get_variance()

    def get_integral_image(self):
        return self.int_img, self.variance

    def calc(self, x, y, sq=False):
        # Calculate value of each pixel
        if x == 0 or y == 0:
            return 0
        # if already calc, return value
        if self.memo[x][y] == 1:
            if not sq:
                return self.int_img[x][y]
            else:
                return self.int_img_sq[x][y]
        else:
            # principal equation
            cummulative = self.calc(x-1, y, sq) + self.calc(x, y-1, sq) - self.calc(x-1, y-1, sq)
            if not sq:
                cummulative += self.img[x-1][y-1]
            else:
                cummulative += self.img_sq[x-1][y-1]
            self.memo[x][y] = 1
            return cummulative

    def get(self):
        # Get the integral image with additional rows/cols of 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.int_img[i][j] = self.calc(i, j)
        # Get the squared integral image with additional rows/cols of 0
        self.memo = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.int_img_sq[i][j] = self.calc(i, j, sq=True)

    def get_variance(self):
        N = (self.shape[0] - 1) * (self.shape[1] - 1) # #pixels
        m = self.int_img[-1][-1] / N # mean
        sum_sq = self.int_img_sq[-1][-1] # sum of x^2
        self.variance = (sum_sq / N) - math.pow(m, 2) # remember to sqrt


def get_sum(int_img, top_left, bottom_right):
    # get summed value over a rectangle (attention to the tuple)

    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    # must swap the tuples since the orientation of the coordinate system
    if top_left == bottom_right:
        return int_img[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    
    return int_img[bottom_right] + int_img[top_left] - int_img[bottom_left] - int_img[top_right]


'''
Note that the orientations of the coordiante system are opposite
-------------------> x
|   x1,y1   x2,y1
|   x1,y2   ...
|   ...
|
y

'''
