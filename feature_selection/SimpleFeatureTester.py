import pandas_tutorital as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


UploadFile = "Feature Optimizer Data.csv"
Dataframe = pd.read_csv(UploadFile, sep=',')
PickedFeatures = ['Medu', 'age', 'Medu', 'G3']
DataPicks = Dataframe[PickedFeatures] #always put predictor variable last
TargetVariable = "G3"
X = np.array(DataPicks.drop([TargetVariable], axis=1))
y = np.array(DataPicks[TargetVariable])

##this puts the data into 4 different arrays: x train, x test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
x_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=0) #add in randomstate= a # to stop randomly changing your arrays

MyLinearRegression = linear_model.LinearRegression().fit(x_train, y_train)
print("Score:", MyLinearRegression.score(X_test, y_test))