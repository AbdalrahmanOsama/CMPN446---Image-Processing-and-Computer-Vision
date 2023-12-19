## We have included all imports and functions of common_functions to make it similar to labs, you are allowed to import others. 

#! ############################ Do NOT Remove any import  ###############################

import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from scipy import fftpack
from scipy.signal import convolve2d
from skimage.util import random_noise
from skimage.filters import median, gaussian
from skimage.filters import roberts, sobel, sobel_h, scharr
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening, skeletonize, thin
from skimage import img_as_ubyte

from skimage.io import imsave

import argparse

import sys

from skimage.exposure import histogram


import math

from skimage.filters import sobel_h, sobel, sobel_v, roberts, prewitt
#####################################################################################################


# Show the figures / plots inside the notebook
def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def show_3d_image(img, title):
    fig = plt.figure()
    fig.set_size_inches((12, 8))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, img.shape[0], 1)
    Y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = img[X, Y]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(title)
    plt.show()


def show_3d_image_filtering_in_freq(img, f):
    img_in_freq = fftpack.fft2(img)
    filter_in_freq = fftpack.fft2(f, img.shape)
    filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)

    img_in_freq = fftpack.fftshift(np.log(np.abs(img_in_freq)+1))
    filtered_img_in_freq = fftpack.fftshift(
        np.log(np.abs(filtered_img_in_freq)+1))

    show_3d_image(img_in_freq, 'Original Image')
    show_3d_image(filtered_img_in_freq, 'Filtered Image')


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

# Start your work here:


def gamma_correct(img, c, gamma):
    img_corrected = np.zeros(img.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            img_corrected[y][x] = c * (img[y][x]**gamma)
    return img_corrected

# TODO: Work here Don't change function name

def f1(img_path=None, debug=False):
    img = io.imread(img_path)
    #rows = img.shape[0]
    #cols = img.shape[1]

    img = rgb2gray(img)

    #for i in range(1,rows-1):
    #    for j in range(1,cols-1):
    #        
    #        neighbours = np.array([img[i-1,j-1] , img[i-1,j], img[i-1,j+1],
    #                               img[i  ,j-1] , img[i  ,j], img[i  ,j+1],
    #                               img[i+1,j-1] , img[i+1,j], img[i+1,j+1]] )
    #        neighbours.sort()
    #        img[i,j] = neighbours[4]
    img = median(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = 255 - img[i,j]
    img = gamma_correct(img , 1 , 3)
    # Write your code here and return the new image
    return img


# TODO: Work here Don't change function name nor parameters
def f2(img_path=None, debug=False):
    img = io.imread(img_path)
    img = rgb2gray(img)
    if np.amax(img)<=1:
        img = (img*255).astype('uint8')
    img[img<130] = 0
    img[img>0] = 255
    gx = sobel_h(img)
    gy = sobel_v(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = ((gx[i,j]**2)+(gy[i,j]**2))**0.5
    # Write your code here and return the new image
    return img

# TODO: Work here Don't change function name nor parameters


def f3(img_path=None, debug=False):
    img = io.imread(img_path)
    img = f2(img_path)
    #SE = np.ones(20, 20)
    print(img)
    #img = binary_dilation(image=img)
    # Write your code here and return the new image
    return img



#! ############################ Do NOT CHANGE the following code:  ###############################
def main(f_name, tc_name, o_name):

    fn = getattr(sys.modules[__name__], f_name)

    img = fn(tc_name)
    imsave(o_name, img_as_ubyte(img), quality=100) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("function_name", help="Function Name")
    parser.add_argument("test_case_file_path", help="Test Case")
    parser.add_argument("output_file_path", help="Output File Name")
    args = parser.parse_args()

    f_name = args.function_name
    tc_name = args.test_case_file_path
    o_name = args.output_file_path

    main(f_name, tc_name, o_name)
