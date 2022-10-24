import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
import time
import functions
import multiple_hot_encoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from permutation_feature_importance import importance_plotter


# show all the unique values in a column
original_df = pd.read_csv("Data/student-mat.csv")
for column in original_df:
    unique_vals = np.unique(original_df[column])
    nr_values = len(unique_vals)
    if nr_values < 10:
        print('Values for', column.rjust(10), ':', nr_values, '--', unique_vals)
    else:
        print('Values for', column.rjust(10), ':', nr_values)

importance_plotter()




