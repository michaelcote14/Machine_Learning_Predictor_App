import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import Functions

from Variables import *




def GetCurrentCoefficients():
    print("(Feature)",
          "[Current Coefficients Value] *The Coefficient below is the correlators of your current data picks, while the corr method above is the correlators of the entire data set")
    for index, feature in enumerate(DataPicks):
        try:
            print("(", feature, ")", "[", MyLinearRegression.coef_[index], "]")
        except:
            pass

def CurrentModelPredictor(DataToPredict):
    # try:
    CurrentModelsInputPrediction = MyLinearRegression.predict(
        [[DataToPredict]])  # this will predict a G3 based on the inputed G1, G2, studytime, failures, and absences
    CurrentModelAccuracy = MyLinearRegression.score(X_test, y_test)
    print('Current Models Input Prediction:', CurrentModelsInputPrediction, 'Score:', CurrentModelAccuracy,
          'Mean Absolute Error:',
          metrics.mean_absolute_error(y_test, CurrentModelsPredictions), 'R2 Score:',
          r2_score(y_test, CurrentModelsPredictions), "Range:",
          CurrentModelsInputPrediction - CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction, "-",
          CurrentModelAccuracy * 0.01 * CurrentModelsInputPrediction + CurrentModelsInputPrediction)  # for R2 score, higher is better
    # except:
    #     print("Current model failed to predict user input")
    #     pass
CurrentModelPredictor([4, 4, 18])