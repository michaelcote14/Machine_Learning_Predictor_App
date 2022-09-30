import pandas_tutorital as pd  # this is to read in data sheets
import numpy as np  # this is for doing interesting things with numbers
import sklearn  # this is the machine learning module
from sklearn import linear_model
import itertools
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot  # this allows you to make graphs
import \
    pickle  # this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
from matplotlib import style  # this changes the style of your plot's grid

#
# UploadFile = "Data/Feature Optimizer Data.csv"
# Dataframe = pd.read_csv(UploadFile, sep=',')
# PickedFeatures = ['Fedu', 'Medu', 'age', 'G3']
# DataPicks = Dataframe[PickedFeatures]
# TargetVariable = "G3"
# X = np.array(DataPicks.drop([TargetVariable], axis=1, randomstate=0))
# y = np.array(DataPicks[TargetVariable])
# x_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #add in randomstate= a # to stop randomly changing your arrays
#
# MyLinearRegression = linear_model.LinearRegression().fit(x_train, y_train)
# CurrentModelsPredictions = MyLinearRegression.predict(X_test) #predicts all the outputs for the x variables in the x_test dataframe

