import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np

Dataframe = pd.read_csv("MeghaData.csv", sep=',')
print(Dataframe.head)
DataPicks = Dataframe[["AT", "V", "AP", "RH", "PE"]]
TargetVariable = "PE"

X = np.array(DataPicks.drop([TargetVariable], axis=1).values)
y = np.array(DataPicks[TargetVariable].values)


##this puts the data into 4 different arrays: x train, x test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0) #add in randomstate= a # to stop randomly changing your arrays


MyLinearRegression = linear_model.LinearRegression()
MyLinearRegression.fit(x_train, y_train) #creates the regression line

print('Coefficients:', MyLinearRegression.coef_) #this tells the coefficients for each input, aka the impact they have


TargetVariablePrediction = MyLinearRegression.predict([[14.96, 41.76, 1024.7, 73.17]]) #this will predict a G3 based on the inputed G1, G2, studytime, failures, and absences
print('TargetVariablePrediction:', TargetVariablePrediction)

import pickle
pickle_in = open("studentmodel.pickle", "rb") #loads the prediction model into the variable "linear"
picklemodelG3 = pickle.load(pickle_in)
try:
    pickleprediction = picklemodelG3.predict([[14.96, 41.76, 1024.7, 73.17]])
    print('pickleprediction:', pickleprediction)
except:
    pass

#shows how accurate your current regression model is
CurrentModelAccuracy = MyLinearRegression.score(x_test, y_test)
print('Current Model Accuracy:', CurrentModelAccuracy)
try:
    PickleModelAccuracy = picklemodelG3.score(x_test, y_test)
    print("Pickle Model Accuracy:", PickleModelAccuracy)
except:
    pass


MyModelsPredictions = MyLinearRegression.predict(x_test)
print("Do you want to run evaluations on all data? Y=Yes N=No")
IndividualDataDecision = input()
if (IndividualDataDecision == 'Y'):
    for x in range(len(MyModelsPredictions)):
        print(MyModelsPredictions[x], x_test[x], y_test[x])

# #this section scales the data to be more equal
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# scaledX = scale.fit_transform(X)
# print(scaledX)






