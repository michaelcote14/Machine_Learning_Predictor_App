import pandas as pd  # this is to read in data sheets
import numpy as np  # this is for doing interesting things with numbers
import sklearn  # this is the machine learning module
from sklearn import linear_model
import itertools
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot  # this allows you to make graphs
import \
    pickle  # this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
from matplotlib import style  # this changes the style of your plot's grid

Dataframe = pd.read_csv("Feature Optimizer Data.csv", sep=',')
print("Dataframe Columns:", Dataframe.columns)
DataColumns = Dataframe.columns
DataFeatures = DataColumns.drop("G3")
PickedFeatures = ['Fedu', 'Medu', 'age', 'G3']
DataPicks = Dataframe[PickedFeatures]
DataFeaturesList = DataFeatures.tolist()
print(DataFeaturesList)
print("DataFeatures:", DataFeatures)





combinations = 0
for i in DataFeaturesList:
    result = itertools.combinations(DataFeaturesList, DataFeaturesList.index(i)+1)
    print("i:", i)
    for item in result:
        print(item)
        combinations = combinations + 1
print("combinations:", combinations)

TargetVariable = "G3"
X = np.array(DataPicks.drop([TargetVariable], axis=1))
y = np.array(DataPicks[TargetVariable])

##this puts the data into 4 different arrays: x train, x test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
x_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #add in randomstate= a # to stop randomly changing your arrays

MyLinearRegression = linear_model.LinearRegression().fit(x_train, y_train)



