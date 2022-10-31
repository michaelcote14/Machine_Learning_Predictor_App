import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle
import time
from sklearn.model_selection import cross_val_score
import scaler

start_time = time.time()
#ToDo make this easier to understand and try to make it to where you only have to put one input
# you might be able to do this by using means of data
known_data = {'G3': 7, 'G2': 8, 'age': 16}

RunEvalution = 'No'
dataframe = scaler.main_scaler()
data = dataframe[['G3', 'G2', 'G1', 'age', 'goout', 'romantic_yes', 'traveltime', 'paid_yes', 'internet_yes', 'studytime']]

target_variable = 'G3'
X = np.array(data.drop([target_variable], axis=1))
y = np.array(data[target_variable])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
data_to_loc = data.iloc[0]
runner = 0
predictor_list = []
for i in data_to_loc:
    predictor_list.append(data_to_loc[runner])
    runner = runner + 1
del predictor_list[0]
def predictor():
    # this makes it to where you can predict only off of the data you have, the rest is the mean of the data
    data_we_have = {'age': 16, 'G2': 10, 'goout': 50}
    key_lst = list(data_we_have.keys())
    value_lst = list(data_we_have.values())
    index_lst = []
    for index in range(len(key_lst)):
        indexes_to_get = data.columns.get_loc(key_lst[index])
        index_lst.append(indexes_to_get)
    print('index list', index_lst)
    #gets the means for all columns
    predictor_lst = []
    for index, column in enumerate(data.columns):
        #makes a list with all the means
        predictor_lst.append(data[column].mean())
    print(predictor_lst)
    #replace the age mean with 16
    for count, index in enumerate(index_lst):
        print('index', index)
        print('count:', count)
        for value in value_lst:
            print('value', value)
            predictor_lst[index] = value_lst[count] #this replaces both columns with 10
    del predictor_lst[0]
    print(predictor_lst)
    print(data.columns[1:100].tolist())
    CurrentModelsPredictions = MyLinearRegression.predict(X_test)
    CurrentModelsInputPrediction = MyLinearRegression.predict([predictor_lst]) #problem line
    current_cross_val_score = cross_val_score(MyLinearRegression, X, y, cv=10).mean()
    current_normal_score = MyLinearRegression.score(X_test, y_test)
    print(CurrentModelsInputPrediction)

    # except ValueError as e:
    #     print('Error in current model predictor: '
    #           'Features input are not equal to features in data file')
    #     print(e)
    #     pass
    # except Exception as e:
    #     print('Other error in current model predictor:')
    #     print(e)


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


    print(':       Statistic       :               Current Model               :     Pickle Model       ')
    nested_list = [[('Target Prediction: ' + target_variable), CurrentModelsInputPrediction, PickleModelsInputPrediction], ['Accuracy', current_cross_val_score, pickle_cross_val_score],
                   ['Mean Absolute Error',  metrics.mean_absolute_error(y_test, CurrentModelsPredictions), Pickle_Mean_Absolute_Error], ['R2 Score', r2_score(y_test, CurrentModelsPredictions),
                    Pickle_R2_Score]]

# ToDo make this accept larger named targets for the table target prediction
    for item in nested_list:
        print(':', item[0], ' '*(20-len(item[0])), ':', item[1],  ' '*(40-len(str(item[1]))),
              ':', item[2],  ' '*(20-len(str(item[2]))))
    print(': Range                 :', CurrentModelsInputPrediction
          - current_cross_val_score * 0.01 * CurrentModelsInputPrediction,
          '-',
          current_cross_val_score * 0.01 * CurrentModelsInputPrediction
          + CurrentModelsInputPrediction, '              :', PickleModelsInputPrediction
          - pickle_cross_val_score * 0.01 * PickleModelsInputPrediction,
          '-',
          PickleModelsInputPrediction
          + pickle_cross_val_score * 0.01 * PickleModelsInputPrediction)

    print(': Cross-Val Difference  :', current_cross_val_score - current_normal_score, '                     :',
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

if __name__ == '__main__':
    predictor()