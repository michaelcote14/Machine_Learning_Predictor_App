import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from Variables import *



def GetCurrentCoefficients():
    print("(Feature)",
          "[Current Coefficients Value] *The Coefficient below is the correlators of your current data picks, while the corr method above is the correlators of the entire data set")
    for index, feature in enumerate(DataPicks):
        try:
            print("(", feature, ")", "[", MyLinearRegression.coef_[index], "]")
        except:
            pass



def CurrentModelPredictor():
    try:

        PredictorInputDataHolder = [PredictorInputData]
        CurrentModelsInputPrediction = MyLinearRegression.predict([PredictorInputData])  # this will predict a G3 based on the inputed G1, G2, studytime, failures, and absences
        CurrentModelAccuracy = MyLinearRegression.score(X_test, y_test)
        print('Current Models Input Prediction:', CurrentModelsInputPrediction, 'Score:', CurrentModelAccuracy,
              'Mean Absolute Error:',
              metrics.mean_absolute_error(y_test, CurrentModelsPredictions), 'R2 Score:',
              r2_score(y_test, CurrentModelsPredictions), "Range:",
              CurrentModelsInputPrediction - CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction, "-",
              CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction + CurrentModelsInputPrediction)  # for R2 score, higher is better
    except:
        print("Current model failed to predict user input")
        pass
CurrentModelPredictor()

def PickleModelPredictor():
    try:
        PickledRegressionLine = pickle.load(
            open("Data/studentmodel.pickle", "rb"))  # loads the prediction model into the variable "linear"
        PickledRegressionLinePredictions = PickledRegressionLine.predict(X_test)
        PickleModelsInputPrediction = PickledRegressionLine.predict([[PicklePredictorInputData]])
        PickleModelAccuracy = PickledRegressionLine.score(X_test, y_test)
        print('Pickle Models Input Prediction:', PickleModelsInputPrediction, 'Score:', PickleModelAccuracy,
              'Mean Absolute Error:', metrics.mean_absolute_error(y_test, PickledRegressionLinePredictions),
              'R2 Score:',
              r2_score(y_test, PickledRegressionLinePredictions), "Range:",
              PickleModelsInputPrediction - PickleModelAccuracy * 0.01 * PickleModelsInputPrediction, "-",
              PickleModelsInputPrediction + PickleModelAccuracy * 0.01 * PickleModelsInputPrediction)
    except:
        print('pickle error')
        pass

def EvaluationRunner():
    Sum, Max = 0, 0
    if (RunEvalution == 'Yes'):
        print("Predicted         [Actual Data]  Actual Score       Difference")
        for x in range(len(CurrentModelsPredictions)):
            print(CurrentModelsPredictions[x], ",", X_test[x], ",", y_test[x], "," "            Difference:",
                  y_test[x] - CurrentModelsPredictions[x])
            IndividualDifference = abs(y_test[x] - CurrentModelsPredictions[x])
            Sum = Sum + IndividualDifference
            if IndividualDifference > Max:
                Max = IndividualDifference
    else:
        pass
    print('Current Models Average Difference On All Data:', Sum / len(CurrentModelsPredictions))
    print('Current Models Max Difference On All Data:', Max)  # maybe use an absolute value?

