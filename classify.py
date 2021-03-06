# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:12:32 2020

@author: vhviv
"""
# %%Imports
from classifier import Classifier
from utils import abs_path, check_folder
import joblib

# Read data
cf = Classifier(input_file='characteristics.csv')

# %%Model validation
#val = cf.validation(batch_size=[16, 20, 24], epochs=[250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 2000], units=[150, 180, 200, 220, 250, 300, 325], cv=10)
#val = cf.validation(batch_size=[16], epochs=[250], units=[150], cv=2)

#txt = open("MyFile.txt", "a")
#txt.write("###Best_params")
#txt.write(str(val.best_params_))
#txt.write("\n\n###Best_index")
#txt.write(str(val.best_index_))
#txt.write("\n\n###Best_score")
#txt.write(str(val.best_score_))
#txt.close()

# %%Model generate table
val = cf.validation(batch_size=[20], epochs=[1000], units=[300], cv=10, save_path='result_table.csv')

# %%Model train
# cf.fit(logs_folder=abs_path("logs\\"),
#        export_dir=abs_path('teste/'))

# %%Read model
#cf = Classifier(import_model=abs_path('teste/save_2020_06_24-17_35_07.h5'))

# %%Results
# Comando para executar Tensorboard
# tensorboard --logdir logs/
# print(history.history.keys())
# %%
