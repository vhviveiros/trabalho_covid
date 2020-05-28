# %%Imports
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
from image_segmentation import segmentate_images as seg
import pandas as pd

covid_path = os.path.join('dataset/covid')
covid_masks_path = os.path.join('cov_masks')

non_covid_path = os.path.join('dataset/normal')
non_covid_masks_path = os.path.join('non_cov_masks')


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# %%Segmentation
check_folder(covid_masks_path)
check_folder(non_covid_masks_path)

seg(covid_path, 'cov_masks')
seg(non_covid_path, 'non_cov_masks')

# %%Read images


def readImages(path):
    return [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(path + "/*g")]


with ThreadPoolExecutor() as executor:
    covid_images = executor.submit(readImages, covid_path)
    covid_masks = executor.submit(readImages, covid_masks_path)

    non_covid_images = executor.submit(readImages, non_covid_path)
    non_covid_masks = executor.submit(readImages, non_covid_masks_path)

# %%Processing


def process(images, masks):
    list = []
    with multiprocessing.Pool() as pool:
        list.append(pool.map(job.job, np.swapaxes([images, masks], 0, 1)))
    return np.squeeze(np.asarray(list))


with ThreadPoolExecutor() as executor:
    cov_processed = executor.submit(
        process, covid_images.result(), covid_masks.result())
    non_cov_processed = executor.submit(
        process, non_covid_images.result(), non_covid_masks.result())

# %%Saving


def save_images(images, save_path):
    for i in range(0, len(images)):
        cv2.imwrite(save_path + '/img' + str(i) + '.png', images[i])


cov_save_path = os.path.join('cov_processed')
non_cov_save_path = os.path.join('non_cov_processed')

check_folder(cov_save_path)
check_folder(non_cov_save_path)

save_images(cov_processed.result(), cov_save_path)
save_images(non_cov_processed.result(), non_cov_save_path)

# %%Saving characteristics


count = 0


def save_characteristics(cov_images, non_cov_images):
    data_size = len(cov_images) + len(non_cov_images)
    # 255 = 254 from histogram + 1 of covid/non-covid
    data = np.zeros((data_size, 255))

    def fill_with(images, type):
        global count
        for img in images:
            hist = np.squeeze(cv2.calcHist([img], [0], None, [254], [1, 255]))
            data[count] = np.append(hist, type)
            count += 1

    fill_with(non_cov_images, 0)
    fill_with(cov_images, 1)

    pd.DataFrame(data).to_csv(os.path.join('characteristics.csv'))


save_characteristics(cov_processed.result(), non_cov_processed.result())

# %%Generating histogram


def save_histogram(args):
    images, save_path = args
    check_folder(save_path)

    for i in range(0, len(images)):
        plt.figure()
        histg = cv2.calcHist([images[i]], [0], None, [254], [
                             1, 255])  # calculating histogram
        plt.plot(histg)
        plt.savefig(save_path + '/img' + str(i) + '.png')
        plt.close()


cov_histograms_path = os.path.join('non_cov_histograms')
non_cov_histograms_path = os.path.join('cov_histograms')

with ThreadPoolExecutor() as executor:
    executor.map(save_histogram, [[non_cov_processed.result(), non_cov_histograms_path], [
                 cov_processed.result(), cov_histograms_path]])