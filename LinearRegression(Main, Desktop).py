import pandas_tutorital as pd
import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import Functions


UploadFile = "Feature Optimizer Data.csv"
Dataframe = pd.read_csv(UploadFile, sep=',')
print("All Correlations\n", Dataframe.corr()['G3']) #shows the correlations of all data
PickedFeatures = ['Fedu', 'Medu', 'age', 'G3']
DataPicks = Dataframe[PickedFeatures] #always put predictor variable last
TargetVariable = "G3"
X = np.array(DataPicks.drop([TargetVariable], axis=1))
y = np.array(DataPicks[TargetVariable])

##this puts the data into 4 different arrays: x train, x test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
x_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #add in randomstate= a # to stop randomly changing your arrays

MyLinearRegression = linear_model.LinearRegression().fit(x_train, y_train)

print("(Feature)",
      "[Coefficient Value] *The Coefficient below is the correlators of your current data picks, while the corr method above is the correlators of the entire data set")
for index, feature in enumerate(DataPicks):
    try:
        print("(", feature, ")", "[", MyLinearRegression.coef_[index], "]")
    except:
        pass

CurrentModelsPredictions = MyLinearRegression.predict(X_test) #predicts all the outputs for the x variables in the x_test dataframe


try:
    CurrentModelsInputPrediction = MyLinearRegression.predict([[4, 4, 18]])  # this will predict a G3 based on the inputed G1, G2, studytime, failures, and absences
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




import pickle
try:
    PickledRegressionLine = pickle.load(
        open("studentmodel.pickle", "rb"))  # loads the prediction model into the variable "linear"
    PickledRegressionLinePredictions = PickledRegressionLine.predict(X_test)
    PickleModelsInputPrediction = PickledRegressionLine.predict([[4, 4, 18]])
    PickleModelAccuracy = PickledRegressionLine.score(X_test, y_test)
    print('Pickle Models Input Prediction:', PickleModelsInputPrediction, 'Score:', PickleModelAccuracy, 'Mean Absolute Error:', metrics.mean_absolute_error(y_test, PickledRegressionLinePredictions), 'R2 Score:',
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
        print(CurrentModelsPredictions[x], ",",  X_test[x], ",",  y_test[x], "," "            Difference:", y_test[x]-CurrentModelsPredictions[x])
        IndividualDifference = abs(y_test[x] - CurrentModelsPredictions[x])
        Sum = Sum+IndividualDifference
        if IndividualDifference > Max:
            Max = IndividualDifference
else:
    pass
print('Current Models Average Difference On All Data:', Sum/len(CurrentModelsPredictions))
print('Current Models Max Difference On All Data:', Max) #maybe use an absolute value?
print("Score:", MyLinearRegression.score(X_test, y_test))


# #this section scales the data to be more equal
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# scaledX = scale.fit_transform(X)
# print(scaledX)

#ToDo make it to where you can use regression on variables that aren't in number format, code for doing this is below
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
#
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
# print(X)



