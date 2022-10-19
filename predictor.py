import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle
import time
import one_hot_encoder_multiple_categories
from cross_validation import crosser

start_time = time.time()

RunEvalution = 'Yes'
dataframe = one_hot_encoder_multiple_categories.encoded_df
data = dataframe[['G3', 'G2', 'G1', 'age', 'goout', 'romantic_yes', 'traveltime', 'paid_yes', 'internet_yes', 'studytime']]

target_variable = 'G3'
X = np.array(data.drop([target_variable], axis=1))
y = np.array(data[target_variable])

PredictorInputData = [4, 4, 4, 4, 4, 4, 4, 4, 4]
PicklePredictorInputData = [4, 4, 4, 4, 4, 4, 4, 4, 4]

# ToDo make all scores the cross value score, because the cross value is more accurate

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)

best_cross_amount, my_cross_val_score_best = crosser(MyLinearRegression, 10)
print('\nbest cross amount:', best_cross_amount)
print('my_cross_val_score_best:', my_cross_val_score_best, '\n')



print('Feature'.center(20, '-'), '  [Current Coefficient Value]') #*The Coefficient below is the correlators '
    #'of your current data picks, while the corr method above is the correlators of the entire data set',

for index, feature in enumerate(data):
    try:
        print(feature.ljust(22), '[', MyLinearRegression.coef_[index], ']')
    except:
      pass
print('\n')

# ToDo make this program use the scaler score
CurrentModelsPredictions = MyLinearRegression.predict(X_test)
print('Current Models Predictions:', CurrentModelsPredictions)
try:
    PredictorInputDataHolder = [PredictorInputData]
    print('Predictor Input Data Holder:', PredictorInputDataHolder)
    CurrentModelsInputPrediction = MyLinearRegression.predict([PredictorInputData])
    print('Current Models Input Prediction:', CurrentModelsInputPrediction)
    CurrentModelAccuracy = MyLinearRegression.score(X_test, y_test)
    print('Current Model Accuracy:', CurrentModelAccuracy)





    print('Current Models Input Prediction for', target_variable, ':',CurrentModelsInputPrediction,
    '\nScore:', CurrentModelAccuracy, '\nMean Absolute Error:',metrics.mean_absolute_error(y_test, CurrentModelsPredictions),
    '\nR2 Score:', r2_score(y_test, CurrentModelsPredictions), '\nRange:',CurrentModelsInputPrediction
    - CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction, '-', CurrentModelAccuracy * 0.01
    * CurrentModelsInputPrediction + CurrentModelsInputPrediction)
except ValueError as e:
    print('Error in current model predictor: '
          'Features input are not equal to features in data file')
    print(e)
    pass
except Exception as e:
    print('Other error in current model predictor:')
    print(e)







#try:
PickledRegressionLine = pickle.load(
    open('Data/studentmodel.pickle', 'rb')
)  # loads the prediction model into the variable 'linear'
PickledRegressionLinePredictions = PickledRegressionLine.predict(X_test)
PickleModelsInputPrediction = PickledRegressionLine.predict(
    [PicklePredictorInputData]) # this is the line that is wrong
PickleModelAccuracy = PickledRegressionLine.score(X_test, y_test)
print(
    '\nPickle Models Input Prediction:',
    PickleModelsInputPrediction,
    '\nScore:',
    PickleModelAccuracy,
    '\nMean Absolute Error:',
    metrics.mean_absolute_error(y_test, PickledRegressionLinePredictions),
    '\nR2 Score:',
    r2_score(y_test, PickledRegressionLinePredictions),
    '\nRange:',
    PickleModelsInputPrediction
    - PickleModelAccuracy * 0.01 * PickleModelsInputPrediction,
    '-',
    PickleModelsInputPrediction
    + PickleModelAccuracy * 0.01 * PickleModelsInputPrediction, '\n'
)
# except ValueError as e:
#     print('\nError in pickle model predictor: '
#           'Pickle input features are not equal to features in data file')
#     print(e)
#     pass
# except Exception as e:
#     print('Other error in pickle model predictor:')
#     print(e)


# mean absolute error is best metric to use
try:
    print(':       Statistic       :    Current Model    :       Pickle Model       :')
    nested_list = [['Input Prediction', CurrentModelsInputPrediction, PickleModelsInputPrediction], ['Score', CurrentModelAccuracy, PickleModelAccuracy],
                   ['Mean Absolute Error',  metrics.mean_absolute_error(y_test, CurrentModelsPredictions),
                    metrics.mean_absolute_error(y_test, PickledRegressionLinePredictions)], ['R2 Score', r2_score(y_test, CurrentModelsPredictions),
                    r2_score(y_test, PickledRegressionLinePredictions)], ['Range', CurrentModelsInputPrediction
            - CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction,
            '-',
            CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction
            + CurrentModelsInputPrediction,  PickleModelsInputPrediction
            - PickleModelAccuracy * 0.01 * PickleModelsInputPrediction,
            '-',
            PickleModelsInputPrediction
            + PickleModelAccuracy * 0.01 * PickleModelsInputPrediction]]
    for item in nested_list:
        print(':', item[0], ' '*(20-len(item[0])), ':', item[1],  ' '*(18-len(str(item[1]))),
              ':', item[2],  ' '*(20-len(str(item[2]))))
except Exception as e:
    print('\nTable failed because of pickle or current model.')
    print(e, '\n')





Sum, Max = 0, 0
if RunEvalution == 'Yes':
    print('Predicted         [Actual Data]  Actual Score', 'Difference'.rjust(70))
    for x in range(len(CurrentModelsPredictions)):
        print(CurrentModelsPredictions[x],X_test[x], y_test[x],'Difference:'.center(80), #y test x is what is wrong
        y_test[x] - CurrentModelsPredictions[x])
        IndividualDifference = abs(y_test[x] - CurrentModelsPredictions[x])
        Sum = Sum + IndividualDifference
        if IndividualDifference > Max:
            Max = IndividualDifference
    print(
        '\nCurrent Models Average Difference On All Data:',
        Sum / len(CurrentModelsPredictions),
    )
    print(
        'Current Models Max Difference On All Data:', Max)
    print('Current Models Range', CurrentModelsInputPrediction
        - CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction,
        '-',
        CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction
        + CurrentModelsInputPrediction)
else:
    pass


# #this section scales the data to be more equal
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# scaledX = scale.fit_transform(X)
# print(scaledX)

# ToDo make it to where you can use regression on variables that aren't in number format,
#  code for doing this is below


print('Runtime:', (time.time() - start_time), 'seconds')