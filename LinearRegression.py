import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np

Dataframe = pd.read_csv("student-mat.csv", sep=';')
DataPicks = Dataframe[["G1", "G2", "G3", "studytime", "failures", "absences"]]
TargetVariable = "G3"
print(DataPicks.describe())

X = np.array(DataPicks.drop([TargetVariable], axis=1))
y = np.array(DataPicks[TargetVariable])
print("Below Data Are All Correlations")
print(DataPicks.corr())


##this puts the data into 4 different arrays: x train, x test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=0) #add in randomstate= a # to stop randomly changing your arrays


MyLinearRegression = linear_model.LinearRegression()
MyLinearRegression.fit(x_train, y_train) #creates the regression line

#see if you can get coefficients to be prefaced by what they represent
print('Coefficients:', MyLinearRegression.coef_) #this tells the coefficients for each input, aka the impact they have
print('y-intercept:', MyLinearRegression.intercept_)

TargetVariablePrediction = MyLinearRegression.predict([[10, 13, 1, 0, 6]]) #this will predict a G3 based on the inputed G1, G2, studytime, failures, and absences
CurrentModelAccuracy = MyLinearRegression.score(x_test, y_test)
print('Current Model Prediction:', TargetVariablePrediction, 'Accuracy:', CurrentModelAccuracy, "Range:", TargetVariablePrediction-CurrentModelAccuracy*0.01*TargetVariablePrediction, "-", CurrentModelAccuracy*0.01*TargetVariablePrediction+TargetVariablePrediction)

import pickle
pickle_in = open("studentmodel.pickle", "rb") #loads the prediction model into the variable "linear"
PickleTargetVariablePrediction = pickle.load(pickle_in)
try:
    PicklePrediction = PickleTargetVariablePrediction.predict([[10, 13, 1, 0, 6]])
    PickleModelAccuracy = PickleTargetVariablePrediction.score(x_test, y_test)
    print('Pickle Model Prediction:', PicklePrediction, 'Accuracy:', PickleModelAccuracy, "Range:", PicklePrediction-PickleModelAccuracy*0.01*PicklePrediction, "-", PicklePrediction+PickleModelAccuracy*0.01*PicklePrediction)
except:
    pass
Sum = 0
MyModelsPredictions = MyLinearRegression.predict(x_test)
print("Do you want to run evaluations on all data? Y=Yes N=No")
IndividualDataDecision = input()
if (IndividualDataDecision == 'Y'):
    print("Predicted         [Actual Data]  Actual Score       Difference")
    for x in range(len(MyModelsPredictions)):
        print(MyModelsPredictions[x], ",",  x_test[x], ",",  y_test[x], "," "            Difference:", y_test[x]-MyModelsPredictions[x])
        IndividualDifference = abs(y_test[x] - MyModelsPredictions[x])
        Sum = Sum+IndividualDifference
        Max = 0
        if IndividualDifference > Max:
            Max = IndividualDifference
print('Sum Difference:', Sum)
print('Average Difference:', Sum/len(MyModelsPredictions))
print('Max Difference:', Max) #maybe use an absolute value?
print(len(MyModelsPredictions))
# #this section scales the data to be more equal
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# scaledX = scale.fit_transform(X)
# print(scaledX)




