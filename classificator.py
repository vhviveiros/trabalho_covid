# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:12:32 2020

@author: vhviv
"""
# %%Imports
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import datetime
from keras.utils import np_utils
from utils import abs_path, check_folder
from models import deep_model

# %%Read csv
ctcs = pd.read_csv('characteristics.csv')
entries = ctcs.iloc[:, 1:255].values
results = ctcs.iloc[:, 255].values


# %%Split into test and training
X_train, X_test, y_train, y_test = train_test_split(
    entries, results, test_size=0.2, random_state=0)

# %%Normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create a basic model instance
model = deep_model()
date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

# Display the model's architecture
model.summary()

log_dir = abs_path("logs\\") + date_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# Fitting the ANN to the Training set
history = model.fit(X_train, y_train, batch_size=16, epochs=250, verbose=1, use_multiprocessing=True,
                    validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# %%Read model
model.load_weights('save_0.93_0.55.h5')

# %%Salva o treinamento
model.save('save_' + date_time + '.h5')

# %%Results
# Comando para executar Tensorboard
# tensorboard --logdir logs/
# print(history.history.keys())

pred = model.predict_classes(X_test)
test = y_test
matrix = confusion_matrix(pred, test)
print(matrix)


# %%
