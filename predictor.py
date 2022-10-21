import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle
import time
import multiple_hot_encoder
from sklearn.model_selection import cross_val_score


start_time = time.time()

RunEvalution = 'No'
dataframe = multiple_hot_encoder.multiple_encoder()
data = dataframe[['G3', 'G2', 'G1', 'age', 'goout', 'romantic_yes', 'traveltime', 'paid_yes', 'internet_yes', 'studytime']]

target_variable = 'G3'
X = np.array(data.drop([target_variable], axis=1))
y = np.array(data[target_variable])

PredictorInputData = [4, 4, 4, 4, 4, 4, 4, 4, 4]
PicklePredictorInputData = [4, 4, 4, 4, 4, 4, 4, 4, 4]


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)



CurrentModelsPredictions = MyLinearRegression.predict(X_test)


try:
    CurrentModelsInputPrediction = MyLinearRegression.predict([PredictorInputData])
    current_model_accuracy = cross_val_score(MyLinearRegression, X, y, cv=10).mean()

except ValueError as e:
    print('Error in current model predictor: '
          'Features input are not equal to features in data file')
    print(e)
    pass
except Exception as e:
    print('Other error in current model predictor:')
    print(e)



PickledRegressionLine = pickle.load(
    open('Data/studentmodel.pickle', 'rb')
)  # loads the prediction model into the variable 'linear'
PickledRegressionLinePredictions = PickledRegressionLine.predict(X_test)
PickleModelsInputPrediction = PickledRegressionLine.predict(
    [PicklePredictorInputData]) # this is the line that is wrong
pickle_model_accuracy = cross_val_score(PickledRegressionLine, X, y, cv=10).mean()


# mean absolute error is best metric to use
try:
    print(':       Statistic       :               Current Model               :     Pickle Model       ')
    nested_list = [['Target Prediction', CurrentModelsInputPrediction, PickleModelsInputPrediction], ['Accuracy', current_model_accuracy, pickle_model_accuracy],
                   ['Mean Absolute Error',  metrics.mean_absolute_error(y_test, CurrentModelsPredictions),
                    metrics.mean_absolute_error(y_test, PickledRegressionLinePredictions)], ['R2 Score', r2_score(y_test, CurrentModelsPredictions),
                    r2_score(y_test, PickledRegressionLinePredictions)]]
    for item in nested_list:
        print(':', item[0], ' '*(20-len(item[0])), ':', item[1],  ' '*(40-len(str(item[1]))),
              ':', item[2],  ' '*(20-len(str(item[2]))))
    print(': Range                 :', CurrentModelsInputPrediction
          - current_model_accuracy * 0.01 * CurrentModelsInputPrediction,
          '-',
          current_model_accuracy * 0.01 * CurrentModelsInputPrediction
          + CurrentModelsInputPrediction, '              :', PickleModelsInputPrediction
          - pickle_model_accuracy * 0.01 * PickleModelsInputPrediction,
          '-',
          PickleModelsInputPrediction
          + pickle_model_accuracy * 0.01 * PickleModelsInputPrediction)
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
        - current_model_accuracy * 0.01 * CurrentModelsInputPrediction,
        '-',
        current_model_accuracy * 0.01 * CurrentModelsInputPrediction
        + CurrentModelsInputPrediction)
    print('Feature'.center(20, '-'), '  ---Current Coefficient Value---')  # *The Coefficient below is the correlators '
    # 'of your current data picks, while the corr method above is the correlators of the entire data set',

    for index, feature in enumerate(data):
        try:
            print(feature.ljust(22), '[', MyLinearRegression.coef_[index], ']')
        except:
            pass
    print('\n')

else:
    pass



print('\nRuntime:', (time.time() - start_time), 'seconds')