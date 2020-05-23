import cv2
import numpy as np
from PIL import ImageEnhance, Image

# Processing


def enhaceContrast(img, factor):
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(np.copy(img))).enhance(factor))


def removePixels(img):
    new_img = np.copy(img)
    for i in range(0, len(new_img)):
        for j in range(0, len(new_img[i])):
            if new_img[i][j] < 10:
                new_img[i][j] = 254
    return new_img


def job(img):
    print("New job started")
    img = cv2.resize(img, (512, 512))
    img = cv2.equalizeHist(img)
    img = 255 - img
    mask = enhaceContrast(img, 150)
    img -= mask
    #img = removePixels(img)
    print("Job finished")
    return img
