import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



Dataframe = pd.read_csv("student-mat.csv", sep=',')
print("All Correlations\n", Dataframe.corr()['G3']) #shows the correlations of all data
DataPicks = Dataframe[["Medu", "Fedu", "G1", "G2", "studytime", "famrel", "freetime", "traveltime", "failures", "health", "Walc", "Dalc", "G3"]] #always put predictor variable last
TargetVariable = "G3"
X = np.array(DataPicks.drop([TargetVariable], axis=1))
y = np.array(DataPicks[TargetVariable])

##this puts the data into 4 different arrays: x train, x test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #add in randomstate= a # to stop randomly changing your arrays

MyLinearRegression = linear_model.LinearRegression()
MyLinearRegression.fit(x_train, y_train) #creates the regression line


print("Feature", "[Coefficient Value] *The Coefficient below is the correlators of your current data picks, while the corr method above is the correlators of the entire data set")
for index, feature in enumerate(DataPicks):
    try:
        print(feature, "[", MyLinearRegression.coef_[index], "]")
    except:
        pass

CurrentModelsPredictions = MyLinearRegression.predict(x_test) #predicts all the outputs for the x variables in the x_test dataframe
CurrentModelsInputPrediction = MyLinearRegression.predict([[2, 2, 14, 16, 3, 4, 5, 1, 0, 3, 1, 1]]) #this will predict a G3 based on the inputed G1, G2, studytime, failures, and absences
CurrentModelAccuracy = MyLinearRegression.score(x_test, y_test)
print('Current Models Input Prediction:', CurrentModelsInputPrediction, 'Accuracy:', CurrentModelAccuracy, 'Mean Absolute Error:',
      metrics.mean_absolute_error(y_test, CurrentModelsPredictions), 'R2 Score:', r2_score(y_test, CurrentModelsPredictions), "Range:", CurrentModelsInputPrediction-CurrentModelAccuracy*0.01*CurrentModelsInputPrediction, "-", CurrentModelAccuracy*0.01*CurrentModelsInputPrediction+CurrentModelsInputPrediction) #for R2 score, higher is better


import pickle
PickledRegressionLine = pickle.load(open("studentmodel.pickle", "rb")) #loads the prediction model into the variable "linear"
PickledRegressionLinePredictions = PickledRegressionLine.predict(x_test)
try:
    PickleModelsInputPrediction = PickledRegressionLine.predict([[2, 2, 14, 16, 3, 4, 5, 1, 0, 3, 1, 1]])
    PickleModelAccuracy = PickledRegressionLine.score(x_test, y_test)
    print('Pickle Models Input Prediction:', PickleModelsInputPrediction, 'Accuracy:', PickleModelAccuracy, 'Mean Absolute Error:', metrics.mean_absolute_error(y_test, PickledRegressionLinePredictions), 'R2 Score:',
          r2_score(y_test, PickledRegressionLinePredictions), "Range:", PickleModelsInputPrediction-PickleModelAccuracy*0.01*PickleModelsInputPrediction, "-", PickleModelsInputPrediction+PickleModelAccuracy*0.01*PickleModelsInputPrediction)
except:
    print('pickle error')
    pass

print("Run evaluations on all data? y = yes, Enter = no")
IndividualDataDecision = input()
Sum, Max = 0, 0
if (IndividualDataDecision == 'y'):
    print("Predicted         [Actual Data]  Actual Score       Difference")
    for x in range(len(CurrentModelsPredictions)):
        print(CurrentModelsPredictions[x], ",",  x_test[x], ",",  y_test[x], "," "            Difference:", y_test[x]-CurrentModelsPredictions[x])
        IndividualDifference = abs(y_test[x] - CurrentModelsPredictions[x])
        Sum = Sum+IndividualDifference
        if IndividualDifference > Max:
            Max = IndividualDifference
else:
    pass
print('Current Models Average Difference On All Data:', Sum/len(CurrentModelsPredictions))
print('Current Models Max Difference On All Data:', Max) #maybe use an absolute value?


#ToDo learn how to properly scale the date
# #this section scales the data to be more equal
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# scaledX = scale.fit_transform(X)
# print(scaledX)

#ToDo make it to where you can add binary data, such as yes or no
#ToDo make all these programs onto one program



