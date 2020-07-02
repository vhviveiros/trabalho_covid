# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:12:32 2020

@author: vhviv
"""
# %%Imports
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from models import classifier_model
from utils import abs_path, check_folder
import datetime
from classifier import Classifier

# Read data
cf = Classifier(input_file='characteristics.csv')

# Model validation
val = cf.validation(batch_size=[32, 16, 24], epochs=[
                    2, 3, 5], cv=2, save_path=abs_path('teste_.csv'))
#cf.validation(batch_size=[32, 16, 24], epochs=[100, 250, 200, 500])

# Model train
# cf.fit(logs_folder=abs_path("logs\\"),
#        export_dir=abs_path('teste/'), epochs=10)

# %%Read model
cf = Classifier(import_model=abs_path('teste/save_2020_06_24-17_35_07.h5'))

# %%Results
# Comando para executar Tensorboard
# tensorboard --logdir logs/
# print(history.history.keys())
# %%
