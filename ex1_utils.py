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
    im = imReadAndConvert(filename, representation)
    if im is not None:
        helperFun(im)

# helpper function for imDisplay
def helperFun(img: np.ndarray):
    if checkImage(img):
        l = len(img.shape)
        if l == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.show()
    else:
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


def imHistogram(img: np.ndarray) -> np.ndarray:
    """
        This is internal function create a histogram of a given image.
        img  : image in grayscale a.k.a 2D, with ranges [0, 255]
    """
    histo = np.zeros((256),dtype=int)
    for column in img:
        for pixel in column:
            histo[pixel] += 1
    return histo

# 4.4
def hsitogramEqualize(imgOrig: np.ndarray) ->(np.ndarray,np.ndarray,np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if checkImage(imgOrig):
        # Convert any image to grayscale 256, save the original YIQ if its RGB
        leng = len(imgOrig.shape)
        if leng == 3:
            YIQ = transformRGB2YIQ(imgOrig)
            im = YIQ[:, :, 0]
            im: np.ndarray = np.round(im * 255).astype(int)
        else:
            im: np.ndarray = np.round(imgOrig * 255).astype(int)
        histoOrig = imHistogram(im)

        # numbers of pixels in the image
        pixelNum = im.shape[0] * im.shape[1]

        # create lut and cumsum
        cumsum = np.zeros((256), dtype=int)
        lut = np.zeros((256), dtype=int)
        cumsum[0] = histoOrig[0]
        for i in range(1, 256):
            cumsum[i] = cumsum[i - 1] + histoOrig[i]
        lut = np.ceil(cumsum * 255 / pixelNum).astype(int)

        EqualImg = np.zeros(im.shape, dtype=int)

        for x in range(0, EqualImg.shape[0]):
            for y in range(0, EqualImg.shape[1]):
                intensity = im[x][y]
                EqualImg[x][y] = lut[intensity]

        Eq_histogram = imHistogram(EqualImg)
        EqualImg = EqualImg/255
        if leng == 3:
            # convert back to RGB
            YIQ[:, :, 0] = EqualImg
            img = transformYIQ2RGB(YIQ)
            return img, histoOrig, Eq_histogram
        return EqualImg, histoOrig, Eq_histogram
    else:
        return
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

    # Initialize the Q's and Z's
    pixelNum = im.shape[0] * im.shape[1]
    histoOrig = imHistogram(im)
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

    # Create internal function for calculating Q, given Z
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
        delta = 0
        for i in range(nQuant):
            minz = z[i]
            maxz = z[i + 1]
            if minz == 0:
                minc = 0
            else:
                minc = cumsum[minz - 1]

            pixels = cumsum[maxz - 1] - minc
            intensities = range_z[minz:maxz]
            pix = histoOrig[minz:maxz]

            qdelta = ((intensities - q[i]) ** 2)
            qdelta = qdelta @ pix
            delta += qdelta
        return math.sqrt(delta)/ pixelNum

    # Create a quantalized image.
    def quantImg() -> np.ndarray:
        qImg = np.zeros(im.shape)
        for i in range(nQuant):
            select = (im >= z[i]) & (im < z[i + 1])
            qImg[select] = q[i]
        return qImg

    # QImage in arr for Err and QImage.
    z[nQuant] = 256
    arr_qim = []
    arr_errors = []
    for i in range(nIter):
        Z()
        Q()
        if (YIQ is not None):
            qim = YIQ.copy()
            qim[:, :, 0] = (quantImg() / 255)
            qim = transformYIQ2RGB(qim)
        else:
            qim = quantImg()
        arr_qim.append(qim)
        arr_errors.append(Errors())

    return arr_qim, arr_errors