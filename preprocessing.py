# %%Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from utils import abs_path, check_folder
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import pandas as pd
from image import *
import time

covid_path = abs_path('dataset/covid')
covid_masks_path = abs_path('cov_masks')

non_covid_path = abs_path('dataset/normal')
non_covid_masks_path = abs_path('non_cov_masks')

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

print("Processing images\n")
cov_processed = cov_processor.process()
non_cov_processed = non_cov_processor.process()

# %%Saving processed images
cov_save_path = abs_path('cov_processed')
non_cov_save_path = abs_path('non_cov_processed')

check_folder(cov_save_path)
check_folder(non_cov_save_path)

ImageSaver(cov_processed).save_to(cov_save_path)
ImageSaver(non_cov_processed).save_to(non_cov_save_path)

# %%Reading processed images
generator = ImageGenerator()

cov_processed_gen, non_cov_processed_gen = generator.generate_processed_data(
    covid_path, non_covid_path)

cov_processed = list(cov_processed_gen.result())
non_cov_processed = list(non_cov_processed_gen.result())

# %%Saving characteristics
characteristics_file = 'characteristics.csv'
ImageCharacteristics(cov_processed, non_cov_processed).save(
    characteristics_file)

# %%
