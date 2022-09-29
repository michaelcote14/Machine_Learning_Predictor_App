import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle
import time

start_time = time.time()

#ToDo fix input problems

RunEvalution = 'Yes'

dataframe = pd.read_csv('Data/student-mat.csv', sep=',')
data = dataframe[['Fedu', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'G1', 'G2', 'G3']]
print('All Correlations:\n', dataframe.corr()['G3'], '\n')  # showsthecorrelationsofalldata

target_variable = 'G3'
X = np.array(data.drop([target_variable], axis=1))
y = np.array(data[target_variable])

PredictorInputData = [4, 4, 4, 4, 4, 4, 4, 4]
PicklePredictorInputData = [4, 4, 4, 4, 4, 4]

##this puts the data into 4 different arrays: x train, x test, y train, and
# y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1
)  # add in randomstate= a # to stop randomly changing your arrays

MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
data = ['apple', 'banana', 'orange', 'grapes']
# need:
for index, feature in enumerate(data):
    try:
        print(feature.ljust(22), '[', MyLinearRegression.coef_[index], ']', dataframe.corr([index])['G3'])
        print('index:', index)
        print('feature:', feature)
    except:
        print('did not work')
        pass
print('\n')