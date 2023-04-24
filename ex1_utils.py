"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 212345466


# helper function to check the image it will be uses almost in all functions
def checkImage(image: np.ndarray) -> bool:
    length = len(image.shape)
    if length == 2:
        return True
    # check if size of the last dimension is 3
    elif length == 3 and image.shape[-1] == 3:
        return True
    else:
        print("Given input cannot be an image. The shape must be either (height, width) or (height, width, 3).")
        return False

# 4.1
def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
        Reads an image, and returns the image converted as requested
        :param filename: The path to the image
        :param representation: GRAY_SCALE or RGB
        :return: The image object
        """
    try:
        image: np.ndarray = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        if representation == 1:
            return 0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]

        elif representation == 2:
            return image
    except:
        print("Image may not exist!")


#4.2
def imDisplay(filename: str, representation: int) ->None:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = imReadAndConvert(filename, representation)
    if img is not None:
        if checkImage(img):
            l = len(img.shape)
            if l == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.show()
        else:
            return
    else:
        print("Image could not be read.")
        return

#4.3
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    after some test i now know how to multiply image by a matrix.
    """
    if checkImage(imgRGB):
        # we need to multiply the matricies.
        mat = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]])
        return imgRGB @ mat.T
    else:
        return


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    if checkImage(imgYIQ):
        # straight forward the ivnerse of YIQ
        mat = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]])
        mat = np.linalg.inv(mat)
        imRGB = imgYIQ @ mat.T
        return imRGB
    else:
        return

# 4.4
def createHistogram(img: np.ndarray) -> np.ndarray:
    """
    :param img: input grayscale image with ranges [0, 255]
    :return: histogram of the input image
    """
    hist = np.zeros((256), dtype=float)
    for i in range(256):
        hist[i] = np.sum(img == i)
    return hist


def hsitogramEqualize(imgOrig: np.ndarray) ->(np.ndarray,np.ndarray,np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if not checkImage(imgOrig):
        return

    # transformRGB2YIQ or save original YIQ if it's RGB
    if len(imgOrig.shape) == 3:
        YIQ = transformRGB2YIQ(imgOrig)
        im = YIQ[:, :, 0]
    else:
        im = imgOrig

    # Round image values to the nearest integer and scale to 0-255 range
    im = np.round(im * 255).astype(int)

    # Compute the number of pixels in the image and the original histogram
    pixelNum = im.shape[0] * im.shape[1]
    histOrg = createHistogram(im)

    # Compute cumulative sum of the histogram and LUT
    cumsum = np.cumsum(histOrg)
    lut = np.ceil(cumsum * 255 / pixelNum).astype(int)

    # Compute the new image with the equalized histogram using the LUT
    imEq = lut[im]

    # Compute the histogram of the equalized image and normalize it
    histEQ = createHistogram(imEq)
    histEQ = histEQ / pixelNum

    # Scale pixel values of the equalized image back to 0-1 range
    imEq = imEq / 255

    # Convert the equalized image back to RGB if the input image was RGB
    if len(imgOrig.shape) == 3:
        YIQ[:, :, 0] = imEq
        img = transformYIQ2RGB(YIQ)
        return img, histOrg, histEQ

    # Return the equalized grayscale image and the histograms
    return imEq, histOrg, histEQ

#4.5
def quantizeImage(imgOrig: np.ndarray, nQuant: int, nIter: int) ->(List[np.ndarray],List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if checkImage(imgOrig) is False:
        return
    # convert to YIQ if its RGB and use the Y channel, otherwise use the grayscale.
    leng = len(imgOrig.shape)
    # YIQ image
    YIQ = None
    if leng == 3:
        YIQ = transformRGB2YIQ(imgOrig)
        im = YIQ[:, :, 0]
        im: np.ndarray = np.ceil(im * 255).astype(int)
    else:
        im: np.ndarray = np.ceil(imgOrig * 255).astype(int)

    pixelNum = im.shape[0] * im.shape[1]
    histoOrig = createHistogram(im)
    # Initialize Z
    z = np.zeros(nQuant + 1, dtype=int)
    # Initialize q
    q = np.zeros(nQuant, dtype=int)
    z[0] = 0
    z[nQuant] = 255
    # Create cumsum
    cumsum = np.zeros((256), dtype=int)
    cumsum[0] = histoOrig[0]
    for i in range(1, 256):
        cumsum[i] = cumsum[i - 1] + histoOrig[i]
    j = 1
    tot = pixelNum / nQuant
    for i in range(1, 256):
        if cumsum[i] > tot * j:
            z[j] = i
            j += 1
            if j == nQuant:
                break
    range_z = np.arange(256)

    # calcualte Z values
    def Z():
        for i in range(1, nQuant - 1):
            z[i] = (q[i] + q[i - 1]) / 2

    # calculating Q
    def Q():
        for i in range(nQuant):
            minz = z[i]
            maxz = z[i + 1]

            if i + 1 == nQuant:
                maxz = 256
            if minz == 0:
                minc = 0
            else:
                minc = cumsum[minz - 1]

            pixels = cumsum[maxz - 1] - minc
            pix = histoOrig[minz:maxz]
            intensities = range_z[minz:maxz]
            q[i] = (pix @ intensities) / pixels

    def Errors() -> float:
        error = 0
        for i in range(nQuant):
            start = z[i]
            end = z[i + 1]

            if start != 0:
                minc = cumsum[start - 1]
            else:
                minc = 0
            pixels = cumsum[end - 1] - minc
            intensities = range_z[start:end]
            histogram = histoOrig[start:end]

            squared_error = ((intensities - q[i]) ** 2) @ histogram
            error += squared_error
        return math.sqrt(error) / pixelNum

    # quantalized image function
    def quanFunc() -> np.ndarray:
        qImg = np.zeros(im.shape)
        for i in range(nQuant):
            select = (im >= z[i]) & (im < z[i + 1])
            qImg[select] = q[i] if i < nQuant - 1 else q[i - 1]
        return qImg

    # quantalized image in arr for Err and QImage.
    qImageArr = []
    errorsArr = []

    z[nQuant] = 256
    for i in range(nIter):
        Z()
        Q()
        qim = YIQ.copy() if YIQ is not None else quanFunc()
        qim[:, 0] = (qim[:, 0] / 255)
        qim = transformYIQ2RGB(qim) if YIQ is not None else qim
        qImageArr.append(qim)
        errorsArr.append(Errors())

    return qImageArr, errorsArr
