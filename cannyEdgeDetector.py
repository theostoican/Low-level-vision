import numpy as np
import math
from skimage import io, viewer, color
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage, misc
from scipy.ndimage import sobel, generic_gradient_magnitude, generic_filter
import matplotlib.pyplot as plt
import queue


from filters import gaussianKernel, convolve2d

def to_ndarray(img):
    im = misc.imread(img, flatten=True)
    im = im.astype('int32')
    return im


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

def gs_filter(img, sigma):
    """ Step 1: Gaussian filter
    Args:
        img: Numpy ndarray of image
        sigma: Smoothing parameter
    Returns:
        Numpy ndarray of smoothed image
    """
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        return gaussian_filter(img, sigma)


def gradient_intensity(img):
    """ Step 2: Find gradients
    Args:
        img: Numpy ndarray of image to be processed (denoised image)
    Returns:
        G: gradient-intensed image
        D: gradient directions
    """

    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    # return the hypothenuse of (Ix, Iy)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return (G, D)


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
            orientation[i][j] = math.atan2(derivImgY[i][j], derivImgX[i][j])

    return magnitude, orientation

def nonMaximumSuppression(magnitude, orientation):
    output = np.zeros((magnitude.shape[0], magnitude.shape[1]))

    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            angle = round_angle(orientation[i][j])

            try:
                if angle == 0 and magnitude[i][j] >= magnitude[i][j] and \
                        magnitude[i][j] >= magnitude[i][j + 1]:
                    output[i][j] = magnitude[i][j]
                elif angle == 45 and magnitude[i][j] >= magnitude[i-1][j+1] and \
                        magnitude[i][j] >= magnitude[i+1][j-1]:
                    output[i][j] = magnitude[i][j]
                elif angle == 90 and magnitude[i][j] >= magnitude[i-1][j] and \
                        magnitude[i][j] >= magnitude[i+1][j]:
                    output[i][j] = magnitude[i][j]
                elif angle == 135 and magnitude[i][j] >= magnitude[i-1][j-1] and \
                        magnitude[i][j] >= magnitude[i+1][j+1]:
                    output[i][j] = magnitude[i][j]
                
            except IndexError as e:
                pass

    return output

def checkValidSize(sizeI, sizeJ, i, j):
    return i < sizeI and i >= 0 and j < sizeJ and j >= 0

def threshold(suprImg, lowT, highT):
    
    strongI, strongJ = np.where(suprImg >= highT)
    suprImg[strongI, strongJ] = 255

    q = queue.Queue()
    for i,j in zip(strongI, strongJ):
        q.put((i, j))

    while not q.empty():
        (i, j) = q.get()
        if checkValidSize(suprImg.shape[0], suprImg.shape[1], i + 1, j):
            if suprImg[i+1][j] >= lowT and suprImg[i+1][j] != 255:
                print("YEs")
                suprImg[i+1][j] = 255
                q.put((i+1, j))
    
    print(suprImg)
    zeroI, zeroJ = np.where(suprImg != 255)
    suprImg[zeroI, zeroJ] = 0

    return suprImg



if __name__ == "__main__":
    img = to_ndarray('small.jpeg')
    #img = color.rgb2gray(img)

    print("DA")
    print (np.max(img))

    derivImgX, derivImgY = derivGaussFilter(img)

    magnitude, orientation = magnitudeAndOrientation(derivImgX, derivImgY)

    #gaussImg = gs_filter(img, 3)

    #(magnitude, orientation) = gradient_intensity(gaussImg)

    suprImg = nonMaximumSuppression(magnitude, orientation)

    edgesImg = threshold(suprImg, 4, 50)

    #plt.imshow(suprImg, cmap=plt.cm.gray)
    #plt.show()
    plt.imshow(edgesImg, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()