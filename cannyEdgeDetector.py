import numpy as np
import math
from skimage import io, viewer, color
import matplotlib.pyplot as plt


from filters import gaussianKernel, convolve2d

def round_angle(angle):
    """ Input angle must be \in [0,180) """
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle


# Canny edge detector functions:
def derivGaussFilter(img):
    gaussFilter = gaussianKernel(7)
    sobelFilterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelFilterY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    derivGaussX = convolve2d(gaussFilter, sobelFilterX)
    derivGaussY = convolve2d(gaussFilter, sobelFilterY)

    derivImgX = convolve2d(img, derivGaussX)
    derivImgY = convolve2d(img, derivGaussY)

    return derivImgX, derivImgY

def magnitudeAndOrientation(derivImgX, derivImgY):
    magnitude = np.zeros_like(derivImgX)
    print (derivImgX.shape)
    orientation = np.zeros((derivImgX.shape[0], derivImgY.shape[1]))

    for i in range(derivImgX.shape[0]):
        for j in range(derivImgX.shape[1]):
            magnitude[i][j] = math.sqrt(derivImgX[i][j] ** 2 + derivImgY[i][j] ** 2)

    return magnitude


    

def nonMaximumSuppression():
    pass

def threshold():
    pass


if __name__ == "__main__":
    img = io.imread('small.jpeg')
    img = color.rgb2gray(img)
    print(img.shape)

    derivImgX, derivImgY = derivGaussFilter(img)

    magnitude = magnitudeAndOrientation(derivImgX, derivImgY)
    print(magnitude.shape)
    print(img.shape)
    plt.imshow(magnitude, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()