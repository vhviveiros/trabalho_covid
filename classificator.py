# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:12:32 2020

@author: vhviv
"""
# %%Imports
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

import datetime
import glob
import cnn
import unet
import fcnn
from sklearn.model_selection import train_test_split
import keras

# %%Read csv
ctcs = pd.read_csv('characteristics.csv')
entries = ctcs.iloc[:, 1:255].values
results = ctcs.iloc[:, 255].values

# %%Read images
covid_path = os.path.join('cov_processed')
non_covid_path = os.path.join('non_cov_processed')


def readImages(path):
    images = []
    for file in glob.glob(path + "/*g"):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))
        images.append(img)
    return images


with ThreadPoolExecutor() as executor:
    covid_images = executor.submit(readImages, covid_path)
    non_covid_images = executor.submit(readImages, non_covid_path)

entries = np.concatenate((covid_images.result(), non_covid_images.result()))
entries = np.repeat(entries[..., np.newaxis], 3, -1)

cov_len = len(covid_images.result())
non_cov_len = len(non_covid_images.result())
results_len = cov_len + non_cov_len
results = np.zeros((results_len))

results[0:cov_len] = 1


# %%Split into test and training
X_train, X_test, y_train, y_test = train_test_split(
    entries, results, test_size=0.2, random_state=0)

# %%Normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%Create FCNN model
model = fcnn.model()

# %%Create CNN model
model = cnn.model()

# %%Create U-NET model
model = unet.model()

# %%Mostra a arquitetura de rede neural desenvolvida
model.summary()

# %%tensorboard configuration
log_dir = "cnn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# %%Insere a base de dados na rede neural proposta e realiza o treinamento
history = model.fit(X_train, y_train, batch_size=32, epochs=25, use_multiprocessing=True,
                    verbose=1, validation_data=(X_test, y_test))

# %%Read model
model.load_weights('save_0.93_0.55.h5')

# %%Salva o treinamento
model.save('save_0.93_0.55.h5')

# %%Results
# Comando para executar Tensorboard
# tensorboard --logdir logs/
# print(history.history.keys())

pred = model.predict_classes(X_test)
test = y_test
matrix = confusion_matrix(pred, test)
print(matrix)


# %%
