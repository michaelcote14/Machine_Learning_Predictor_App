import pandas as pd  # this is to read in data sheets
import numpy as np  # this is for doing interesting things with numbers
import sklearn  # this is the machine learning module
from sklearn import linear_model
import itertools
from runtime_calculator import iterator_runtime_predictor
import functions
import time

dataframe = pd.read_csv('C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Data/student-mat(Numerical Only).csv', sep=',')
data = dataframe[['Medu', 'Fedu', 'G1', 'G2', 'studytime', 'famrel', 'G3']]

print("All dataframe Columns:", dataframe.columns)
AllDataColumns = dataframe.columns
AlldataframesColumnsList = AllDataColumns.tolist()
DataPicks = dataframe[AlldataframesColumnsList]
PickeddataframeColumns = AllDataColumns.drop("G3")
PickeddataframeColumnsList = PickeddataframeColumns.tolist()
newdata = dataframe[PickeddataframeColumnsList]

TargetVariable = "G3"

best = 0
combinations = 0
run_throughs = 100
iterator_runtime_predictor(run_throughs)
start_time = time.time()



# functions.line_checker('best_data', 12)



print('\nRun feature iterator? Hit ENTER for yes')
user_input = input()
if user_input == '':
    pass
else:
    quit()


total_score = 0
for loop in PickeddataframeColumnsList:
    result = itertools.combinations(PickeddataframeColumnsList, PickeddataframeColumnsList.index(loop)+1)
    for item in result:
        print("item:", list(item))
        for i in range(run_throughs):
            combinations = combinations + 1

            newdata = list(item)

            X = np.array(dataframe[newdata])
            y = np.array(data[TargetVariable])

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)  # add in randomstate= a # to stop randomly changing your arrays

            MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
            print('Score:', MyLinearRegression.score(X_test, y_test))
            # make this add up 5 times first one should equal 0.184
            total_score = total_score + MyLinearRegression.score(X_test, y_test)
            my_linear_regression_score = MyLinearRegression.score(X_test, y_test)

            if my_linear_regression_score > best:
                best = my_linear_regression_score
                print('newdata', newdata)
                best_features = newdata
                print('best_features', best_features)


        # print('\nTotal Score:', total_score)
        average_score = total_score/run_throughs
        print('Average Score:', average_score, '\n')
        total_score = 0
        if average_score > best:
            best = average_score
            best_features = newdata
            print('newdata:', newdata)
            print('best_features:', best_features)


print("Total Combinations:", combinations)
print('Best Score:', best)
print('Best Features:', best_features)


#ToDo fix this bug, its reading in, so the len function won't work
# read in the current best file
print("best_data's Best Score:")
text_best_score = functions.text_file_reader('best_data', 13, 31)


# write to the file
if float(text_best_score) < best:
    text_data_list = ['\n\nBest Score:', str(best), '\nBest Features:',  str(best_features),
    '\nRunthroughs:', str(run_throughs), '\nTime to Run:', str(time.time()-start_time), 'seconds',
    '\nDate Ran:', str(time.asctime())]
    string_data_list = (', '.join(text_data_list))
    functions.text_file_appender('best_data', string_data_list )




