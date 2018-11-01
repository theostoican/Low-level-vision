import numpy as np
import matplotlib.pyplot as plt
from skimage import io, viewer, color

def gaussianKernel(totalSize):
    size = totalSize // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    g = np.exp(-(x**2/float(size)+y**2/float(size)))
    return g / g.sum()

def boxFilter(totalSize):
    size = totalSize // 2

    return np.ones((size, size)) / (totalSize * totalSize)

def convolve2d(image, kernel):
    output = np.zeros_like(image)
    filterSize = kernel.shape[0]
    padSize = filterSize // 2

    image_padded = np.zeros((image.shape[0] + 2 * padSize, image.shape[1] + 2 * padSize))
    image_padded[padSize:-padSize, padSize:-padSize] = image

    print(image.shape)
    print(image_padded.shape)

    for x in range(image.shape[0]):     # Loop over every pixel of the image
        for y in range(image.shape[1]):
            #print(x, y)
            output[x][y] = (kernel * image_padded[x:x + filterSize, y: y + filterSize]).sum()

    return output

if __name__ == "__main__":

    #kernel = gaussianKernel(9)
    
    img = io.imread('small.jpeg')
    img = color.rgb2gray(img)

    # Box filter
    kernel = boxFilter(11)

    blurredImage = convolve2d(img, kernel)

    # Gaussian filter
    #image_sharpen = convolve2d(img, kernel)
    
    # Plot the filtered image
    plt.imshow(blurredImage, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()



    #viewer.ImageViewer(img).show() 

    #plt.imshow(gaussian_kernel_array, cmap=plt.get_cmap('jet'), interpolation='nearest')
    #plt.colorbar()
    #plt.show()