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
import pandas as pd
from image import ImageGenerator, ImageProcessor, ImageSaver, ImageSegmentator
import time

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

ImageSegmentator(folder_in=covid_path,
                 folder_out=covid_masks_path).segmentate()
ImageSegmentator(folder_in=non_covid_path,
                 folder_out=non_covid_masks_path).segmentate()

# %%Read images
generator = ImageGenerator()

covid_images, covid_masks, non_covid_images, non_covid_masks = generator.generate_preprocessing_data(
    covid_path,
    covid_masks_path,
    non_covid_path,
    non_covid_masks_path
)

# %%Processing
cov_processor = ImageProcessor(
    list(covid_images.result()),
    list(covid_masks.result()))

non_cov_processor = ImageProcessor(
    list(non_covid_images.result()),
    list(non_covid_masks.result()))

print("Processing images")
cov_processed = cov_processor.process()
non_cov_processed = non_cov_processor.process()

# with ThreadPoolExecutor() as executor:
#     cov_processed = executor.submit(cov_processor.process)
#     non_cov_processed = executor.submit(non_cov_processor.process)

# %%Saving
cov_save_path = os.path.join('cov_processed')
non_cov_save_path = os.path.join('non_cov_processed')

check_folder(cov_save_path)
check_folder(non_cov_save_path)

ImageSaver(cov_processed).save_to(cov_save_path)
ImageSaver(non_cov_processed).save_to(non_cov_save_path)

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
