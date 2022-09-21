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
from runtime_calculator import iterator_runtime_predictor


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

best = 0
combinations = 0
runtimes = 2
iterator_runtime_predictor(runtimes)


print('\nRun feature iterator? Hit ENTER for yes')
user_input = input()
if user_input == '':
    pass
else:
    quit()

total_score = 0
for loop in PickeddataframeColumnsList:
    result = itertools.combinations(PickeddataframeColumnsList, PickeddataframeColumnsList.index(loop)+1)
    print("loop:", loop)
    average_score = []
    for item in result:
        print("item:", list(item))
        for i in range(runtimes):
            combinations = combinations + 1
            newdata = list(item)
            X = np.array(dataframe[newdata])
            y = np.array(data[TargetVariable])

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)  # add in randomstate= a # to stop randomly changing your arrays

            MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
            print('Score:', MyLinearRegression.score(X_test, y_test))



            MyLinearRegressionScore = MyLinearRegression.score(X_test, y_test)

            #ToDo Fix this bug
            total_score = total_score + MyLinearRegression.score(X_test, y_test)
            print('\nTotal Score:', total_score)
            average_score = total_score / runtimes
            print('\nAverage Score:', average_score)

        #ToDo put in average score calculator, that way you can accurately test feature groups


        #ToDo save the best score/features into a file of some sort, then upload it and only
        #ToDo change the file if the accuracy is greater
            # if MyLinearRegressionScore > best:
            #     best = MyLinearRegressionScore
            #     best_features = newdata


# print("Total Combinations:", combinations)
# print('Best Score:', best)
# print('Best Features:', best_features)

current_best_score = 0.8313653113470161
current_best_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'famrel', 'Dalc', 'health', 'G1', 'G2']






