# Imports
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import job

covid_path = os.path.join('dataset/covid')
non_covid_path = os.path.join('dataset/normal')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

# Read images


def readImages(path):
    return [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(path + "/*g")]


with ThreadPoolExecutor() as executor:
    covid_images = executor.submit(readImages, covid_path)
    non_covid_images = executor.submit(readImages, non_covid_path)

# Processing


def process(images):
    list = []
    with multiprocessing.Pool() as pool:
        list.append(pool.map(job.job, images))
    return np.squeeze(np.asarray(list))


with ThreadPoolExecutor() as executor:
    cov_processed = executor.submit(process, covid_images.result())
    non_cov_processed = executor.submit(process, non_covid_images.result())

# Generating histogram


def histogram(images, count):
    res = []
    plt.figure(count)
    for i in images:
        histg = cv2.calcHist([i], [0], None, [256], [
                             0, 256])  # calculating histogram
        res.append(plt.plot(histg))
    count += 1
    return res


non_cov_histogram = histogram(non_cov_processed, 0)
cov_histogram = histogram(cov_processed, 1)

# Saving


def save_images(images, save_path):
    for i in range(0, len(images)):
        cv2.imwrite(save_path + '/img' + str(i) + '.png', images[i])


cov_save_path = os.path.join('cov_processed')
non_cov_save_path = os.path.join('non_cov_processed')

save_images(cov_processed.result(), cov_save_path)
save_images(non_cov_processed.result(), non_cov_save_path)
