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


dataframe = pd.read_csv('C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Data/student-mat(Numerical Only).csv', sep=',')
data = dataframe[['Medu', 'Fedu', 'G1', 'G2', 'studytime', 'famrel', 'G3']]




print("All dataframe Columns:", dataframe.columns)
AllDataColumns = dataframe.columns
AlldataframesColumnsList = AllDataColumns.tolist()
print("All Data Frame Columns List", AlldataframesColumnsList)
DataPicks = dataframe[AlldataframesColumnsList]
print('Data Picks:\n', DataPicks)

PickeddataframeColumns = AllDataColumns.drop("G3")
print("Picked dataframe Columns:", PickeddataframeColumns, '\n')
PickeddataframeColumnsList = PickeddataframeColumns.tolist()
print("Picked dataframe Columns List", PickeddataframeColumnsList, '\n')
newdata = dataframe[PickeddataframeColumnsList]

print("PickeddataframeColumnsList:", PickeddataframeColumnsList)
print("newdata:\n", newdata)


#PickeddataframeColumnsList: ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
#data needed: matrix of: age  Medu  Fedu  traveltime  studytime  ...  Walc  health  absences  G1  G2

TargetVariable = "G3"

combinations = 0
for loop in PickeddataframeColumnsList:
    result = itertools.combinations(PickeddataframeColumnsList, PickeddataframeColumnsList.index(loop)+1)
    print("loop:", loop)
    for item in result:
        print("item:", list(item))
        combinations = combinations + 1
        newdata = list(item)
        print("newdata:", newdata)
        X = np.array(dataframe[newdata])
        y = np.array(data[TargetVariable])
        print('X shape:', X.shape)
        print('y shape:', y.shape)


                #stuck here right now
        # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1,
        #                                                                             random_state=0)  # add in randomstate= a # to stop randomly changing your arrays
        #
        # MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
        # print(MyLinearRegression.score(X_test, y_test))

print("combinations:", combinations)




#
# print('DataPicks:\n', DataPicks)
# X = np.array(DataPicks.drop([TargetVariable], axis=1))
# y = np.array(DataPicks[TargetVariable])
#
# ##this puts the data into 4 different arrays: X train, X test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=0) #add in randomstate= a # to stop randomly changing your arrays
#
# MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
# print(MyLinearRegression.score(X_test, y_test))



