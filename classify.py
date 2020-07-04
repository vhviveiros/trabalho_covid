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
val = cf.validation(batch_size=[16], epochs=[250], units=[
                    180], cv=10, save_path=abs_path('result_table.csv'))

txt = open("MyFile.txt", "a")
txt.write("###Best_params")
txt.write(str(val.best_params_))
txt.write("\n\n###Best_index")
txt.write(str(val.best_index_))
txt.write("\n\n###Best_score")
txt.write(str(val.best_score_))
txt.write("\n\n###Val")
txt.write(str(val))
txt.write("\n\n###Score")
txt.write(str(val.score))
txt.write("\n\n###cv_results")
txt.write(str(val.cv_results_))
txt.close()

dump_folder = abs_path('results')
check_folder(dump_folder)

joblib.dump(str(val), dump_folder + '/val.joblib')
joblib.dump(str(val.cv_results_), dump_folder + '/cv_results.joblib')
joblib.dump(val.best_params_, dump_folder + '/best_params.joblib')
joblib.dump(val.best_index_, dump_folder + '/best_index.joblib')
joblib.dump(str(val.best_score_), dump_folder + '/best_score.joblib')
joblib.dump(str(val.score), dump_folder + '/score.joblib')

# %%Model generate table
val = cf.validation(batch_size=[32, 16, 24], epochs=[
    50, 100, 200, 250], units=[180, 200, 220], cv=10)

# %%Model train
cf.fit(logs_folder=abs_path("logs\\"),
       export_dir=abs_path('teste/'))

# %%Read model
#cf = Classifier(import_model=abs_path('teste/save_2020_06_24-17_35_07.h5'))

# %%Results
# Comando para executar Tensorboard
# tensorboard --logdir logs/
# print(history.history.keys())
# %%
