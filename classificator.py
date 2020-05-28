# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:12:32 2020

@author: vhviv
"""
# %%Imports
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.engine.sequential import Sequential

# %%Read csv
ctcs = pd.read_csv('characteristics.csv')
entries = ctcs.iloc[:, 1:255].values
result = ctcs.iloc[:, 255].values

# %%Split into test and training
X_train, X_test, y_train, y_test = train_test_split(
    entries, result, test_size=0.2, random_state=0)

# %%Normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%Create FCNN model


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(180, activation='relu', input_shape=(254,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(180, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(180, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(180, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(180, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(180, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['acc', 'mse', 'mae', 'mape'])

    return model


model = create_model()

# %%Mostra a arquitetura de rede neural desenvolvida
model.summary()

# %%Insere a base de dados na rede neural proposta e realiza o treinamento
history = model.fit(X_train, y_train, batch_size=16, epochs=500,
                    verbose=1, validation_data=(X_test, y_test))

# %%Salva o treinamento
model.save('save.h5')
