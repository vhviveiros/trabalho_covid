import cv2
import numpy as np
from PIL import ImageEnhance, Image


def enhaceContrast(img, factor):
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(np.copy(img))).enhance(factor))


def removePixels(img):
    new_img = np.copy(img)
    for i in range(0, len(new_img)):
        for j in range(0, len(new_img[i])):
            if new_img[i][j] < 100:
                new_img[i][j] = 254
    return new_img


def cutImage(img, mask):
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            if mask[i][j] <= 20:
                img[i][j] = 0
    return img


def job(args):
    print("New job started")
    img = cv2.resize(args[0], (512, 512))
    mask = cv2.resize(args[1], (512, 512))

    #img = cv2.equalizeHist(img)

    img = cutImage(img, mask)
    #img = removePixels(img)
    print("Job finished")
    return img
