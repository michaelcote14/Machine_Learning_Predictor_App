import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle


RunEvalution = 'Yes'

dataframe = pd.read_csv('Data/student-mat.csv', sep=',')
data = dataframe[['Medu', 'Fedu', 'G1', 'G2', 'studytime', 'famrel', 'G3']]
print('All Correlations:\n', dataframe.corr()['G3'], '\n')  # showsthecorrelationsofalldata

target_variable = 'G3'
X = np.array(data.drop([target_variable], axis=1))
y = np.array(data[target_variable])

PredictorInputData = [4, 4, 4, 4, 4, 4]
PicklePredictorInputData = [4, 4, 4, 4, 4, 4]

##this puts the data into 4 different arrays: x train, x test, y train, and
# y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1, random_state=0
)  # add in randomstate= a # to stop randomly changing your arrays

MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)



print(
    '(Feature)',
    '[Current Coefficients Value] *The Coefficient below is the correlators '
    'of your current data picks, while the corr method above is the correlators of the entire data set',
)
for index, feature in enumerate(data):
    try:
        print('(', feature, ')', '[', MyLinearRegression.coef_[index], ']')
    except:
      pass
print('\n')

CurrentModelsPredictions = MyLinearRegression.predict(
X_test
)  # predicts all the outputs for the x variables in the x_test DataFrame



try:
    PredictorInputDataHolder = [PredictorInputData]
    CurrentModelsInputPrediction = MyLinearRegression.predict(
        [PredictorInputData]
    )  # this will predict a G3 based on the inputed G1, G2, studytime, failures, and absences
    CurrentModelAccuracy = MyLinearRegression.score(X_test, y_test)
    print(
        'Current Models Input Prediction:',
        CurrentModelsInputPrediction,
        '\nScore:',
        CurrentModelAccuracy,
        '\nMean Absolute Error:',
        metrics.mean_absolute_error(y_test, CurrentModelsPredictions),
        '\nR2 Score:',
        r2_score(y_test, CurrentModelsPredictions),
        '\nRange:',
        CurrentModelsInputPrediction
        - CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction,
        '-',
        CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction
        + CurrentModelsInputPrediction
    )  # for R2 score, higher is better
except ValueError as e:
    print('Error in current model predictor: '
          'Features input are not equal to features in data file')
    print(e)
    pass
except Exception as e:
    print('Other error in current model predictor:')
    print(e)







try:
    PickledRegressionLine = pickle.load(
        open('Data/studentmodel.pickle', 'rb')
    )  # loads the prediction model into the variable 'linear'
    PickledRegressionLinePredictions = PickledRegressionLine.predict(X_test)
    PickleModelsInputPrediction = PickledRegressionLine.predict(
        [PicklePredictorInputData])
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
        + PickleModelAccuracy * 0.01 * PickleModelsInputPrediction,
    )
except ValueError as e:
    print('\nError in pickle model predictor: '
          'Pickle input features are not equal to features in data file')
    print(e)
    pass
except Exception as e:
    print('Other error in pickle model predictor:')
    print(e)


#table
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





Sum, Max = 0, 0
if RunEvalution == 'Yes':
    print('Predicted         [Actual Data]  Actual Score       Difference')
    for x in range(len(CurrentModelsPredictions)):
        print(
            CurrentModelsPredictions[x],
            ',',
            X_test[x],
            ',',
            y_test[x],
            ',' '            Difference:',
            y_test[x] - CurrentModelsPredictions[x],
        )
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
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
#
# ToDo find out why every time I run the current model, the score changes
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
# print(X)
# ToDo install black on laptop


