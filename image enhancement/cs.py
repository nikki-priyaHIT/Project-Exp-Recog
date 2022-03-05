#import numpy,cv2,matplotlip
import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt

# Create an image object
im = '/content/sample_2.jpg'

# Read the image as gray scale image
image = cv.imread(im,cv.IMREAD_GRAYSCALE)

# Display the grayscale image before contrast stretching
cv2_imshow(image)
print("\t")
hist,bin = np.histogram(image,256)
plt.plot(bin[0:-1],hist)
# Display the histogram before contrast stretching
plt.show()

# calculate the maximum and minimum pixel value
lim1=np.min(image)
lim2=np.max(image)
print("\n")

# Display the maximum and minimum pixel value
print("Minimum pixel value :",lim1)
print("Maximum pixel value :",lim2)
print("\t")

# Create an image1 object
# Formula to calculate Contrast stretching
image1 = (image - lim1)/(lim2-lim1)*255

# Display the grayscale image after contrast stretching
cv2_imshow(image1)
print("\t")
hist,bin = np.histogram(image1,256)
plt.plot(bin[0:-1],hist)
# Display the histogram after contrast stretching
plt.show()
