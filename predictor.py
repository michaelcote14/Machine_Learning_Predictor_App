import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle
import time
from sklearn.model_selection import cross_val_score
import scaler
import matplotlib.pyplot as plt
import seaborn as sns

start_time = time.time()
# ToDo fix the predictor :bool object has no attribute 'any'

RunEvalution = 'No'
dataframe = scaler.main_scaler_non_printing()
data = dataframe[['G3', 'G2', 'G1', 'age', 'goout', 'romantic_yes', 'traveltime', 'paid_yes', 'internet_yes', 'studytime']]
print('data\n', data)
target_variable = 'traveltime'
X = np.array(data.drop([target_variable], axis=1), dtype='object')
y = np.array(data[target_variable], dtype='object')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)

def predictor():
    # ToDo scaled data is not accurate to prediction data
    # steps: 1. put prediction data into scaler row?
    # this makes it to where you can predict only off of the data you have, the rest is the mean of the data
    data_we_have = {'age': 16.0, 'G2': 10.0, 'goout': 50, 'internet_yes': 100}
    key_lst = list(data_we_have.keys())
    value_lst = list(data_we_have.values())
    # for values in value_lst:

    index_lst = []
    for index in range(len(key_lst)):
        indexes_to_get = data.columns.get_loc(key_lst[index]) #either this line
        index_lst.append(indexes_to_get)
    #gets the means for all columns
    predictor_lst = []
    for index, column in enumerate(data.columns): #this line
        #makes a list with all the means
        predictor_lst.append(data[column].mean()) #this line
    for count, index in enumerate(index_lst):
        for value in value_lst:
            predictor_lst[index] = value_lst[count]
    del predictor_lst[0]
    print('predictor lst:', predictor_lst)
    #ToDo put predictor list in scaler dataframe
    CurrentModelsPredictions = MyLinearRegression.predict(X_test)
    CurrentModelsInputPrediction = MyLinearRegression.predict([predictor_lst])#problem line
    current_cross_val_score = cross_val_score(MyLinearRegression, X, y, cv=10).mean()
    current_normal_score = MyLinearRegression.score(X_test, y_test)


    try:
        PickledRegressionLine = pickle.load(
            open('Data/studentmodel.pickle', 'rb')
        )  # loads the prediction model into the variable 'linear'
        PickledRegressionLinePredictions = PickledRegressionLine.predict(X_test) # problem line because pickle doesnt work
        PickleModelsInputPrediction = PickledRegressionLine.predict(
            [predictor_lst]) # this is the line that is wrong
        pickle_cross_val_score = cross_val_score(PickledRegressionLine, X, y, cv=10).mean()
        pickle_normal_score = PickledRegressionLine.score(X_test, y_test)
        Pickle_Mean_Absolute_Error = metrics.mean_absolute_error(y_test, PickledRegressionLinePredictions)
        Pickle_R2_Score = r2_score(y_test, PickledRegressionLinePredictions)
    except:
        PickledRegressionLinePredictions, PickleModelsInputPrediction, pickle_cross_val_score = 0, 0, 0
        Pickle_Mean_Absolute_Error, Pickle_R2_Score, pickle_normal_score = 0, 0, 0


    print(':             Statistic                :              Current Model                :        Pickle Model       ')
    nested_list = [[('Target Prediction: ' + target_variable), CurrentModelsInputPrediction, PickleModelsInputPrediction], ['Accuracy', current_cross_val_score, pickle_cross_val_score],
                   ['Mean Absolute Error',  metrics.mean_absolute_error(y_test, CurrentModelsPredictions), Pickle_Mean_Absolute_Error], ['R2 Score', r2_score(y_test, CurrentModelsPredictions),
                    Pickle_R2_Score]]
    for item in nested_list:
        print(':', item[0], ' '*(35-len(item[0])), ':', item[1],  ' '*(40-len(str(item[1]))),
              ':', item[2],  ' '*(20-len(str(item[2]))))
    print(': Range                                :', CurrentModelsInputPrediction
          - current_cross_val_score * 0.01 * CurrentModelsInputPrediction,
          '-',
          current_cross_val_score * 0.01 * CurrentModelsInputPrediction
          + CurrentModelsInputPrediction, '              :', PickleModelsInputPrediction
          - pickle_cross_val_score * 0.01 * PickleModelsInputPrediction,
          '-',
          PickleModelsInputPrediction
          + pickle_cross_val_score * 0.01 * PickleModelsInputPrediction)

    print(': Cross-Val Difference                 :', format(current_cross_val_score - current_normal_score, '.17f'), '                     :',
      pickle_cross_val_score - pickle_normal_score)
    print('Positive number above means cross val score was higher, which means your model is overfitting')



    Sum, Max = 0, 0
    if RunEvalution == 'Yes':
        print('    Predicted                        [Actual Data]                         Actual Score', 'Difference'.rjust(21))
        for x in range(len(CurrentModelsPredictions)):
            print(str(CurrentModelsPredictions[x]).ljust(23),X_test[x], str(y_test[x]).rjust(20),
                  str(y_test[x] - CurrentModelsPredictions[x]).rjust(30))
            IndividualDifference = abs(y_test[x] - CurrentModelsPredictions[x])
            Sum = Sum + IndividualDifference
            if IndividualDifference > Max:
                Max = IndividualDifference


        for index, feature in enumerate(data):
            try:
                print(feature.ljust(22), '[', MyLinearRegression.coef_[index], ']')
            except:
                pass
        print('\n')

    else:
        pass
    #plotter
    plt.figure(figsize=(15, 10))
    print('y_test:', y_test.size)
    print('Current Models Predictions:', CurrentModelsPredictions.size)
    plt.scatter(y_test, CurrentModelsPredictions, c='blue') #problem line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', c='red', lw=3)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual Vs Predicted Values')
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()



if __name__ == '__main__':
    predictor()