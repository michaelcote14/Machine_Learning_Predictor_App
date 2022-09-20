import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import Functions
import pickle
import time


UploadFile = "Feature Optimizer Data.csv"
DataFrame = pd.read_csv(UploadFile, sep=',')
print("AllCorrelations\n",DataFrame.corr()['G3'])#showsthecorrelationsofalldata
print("All DataFrame Columns:", DataFrame.columns)
AllDataColumns = DataFrame.columns
AllDataFramesColumnsList = AllDataColumns.tolist()
print("All Data Frame Columns List", AllDataFramesColumnsList)
DataPicks = DataFrame[AllDataFramesColumnsList]

PickedDataFrameColumns = AllDataColumns.drop("G3")
print("Picked DataFrame Columns:", PickedDataFrameColumns)
PickedDataFrameColumnsList = PickedDataFrameColumns.tolist()
print("Picked DataFrame Columns List", PickedDataFrameColumnsList)

TargetVariable = "G3"
X = np.array(DataPicks.drop([TargetVariable], axis=1, randomstate=0))
y = np.array(DataPicks[TargetVariable])

##this puts the data into 4 different arrays: x train, x test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=1) #add in randomstate= a # to stop randomly changing your arrays

MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)

Functions.GetCurrentCoefficients()

CurrentModelsPredictions = MyLinearRegression.predict(X_test) #predicts all the outputs for the x variables in the x_test DataFrame


Functions.CurrentModelPredictor() #pass in your data that you want predicted in the parenthesis


Functions.PickleModelPredictor()

Functions.EvaluationRunner()



# #this section scales the data to be more equal
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# scaledX = scale.fit_transform(X)
# print(scaledX)

#ToDo make it to where you can use regression on variables that aren't in number format, code for doing this is below
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
#
#ToDo find out why every time I run the current model, the score changes
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
# print(X)



