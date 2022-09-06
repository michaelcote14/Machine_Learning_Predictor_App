import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np

#option 1
#df = pd.read_csv("student-mat.csv", sep=";")
#option 2
Dataframe = pd.read_csv("student-mat.csv", sep=';')
DataPicks = Dataframe[["G1", "G2", "G3", "studytime", "failures", "absences"]]
PredictorVariableG3 = "G3"

#option 1
# X = df[['G1', 'G2', 'studytime', 'failures', 'absences']] #puts all this data for these columns in X for your independant variables
# y = df[['G3']] #puts your dependant variable into y
#option 2
X = np.array(DataPicks.drop([PredictorVariableG3], axis=1))
y = np.array(DataPicks[PredictorVariableG3])


##this puts the data into 4 different arrays: x train, x test, y train, and y test, the random_state parameter chooses how to randomly split the data. not specifying changes it each time.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #add in randomstate= a # to stop randomly changing your arrays


MyLinearRegression = linear_model.LinearRegression()
MyLinearRegression.fit(x_train, y_train) #creates the regression line


predictedG3 = MyLinearRegression.predict([[6, 9, 1, 2, 14]]) #this will predict a G3 based on the inputed G1, G2, studytime, failures, and absences
print('predictedG3:', predictedG3)

import pickle
pickle_in = open("studentmodel.pickle", "rb") #loads the prediction model into the variable "linear"
picklemodelG3 = pickle.load(pickle_in)
pickleprediction = picklemodelG3.predict([[6, 9, 1, 2, 14]])
print('pickleprediction:', pickleprediction)

MyModelsPredictions = MyLinearRegression.predict(x_test)
for x in range(len(MyModelsPredictions)):
    print(MyModelsPredictions[x], x_test[x], y_test[x])

# #this section scales the data to be more equal
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# scaledX = scale.fit_transform(X)
# print(scaledX)


print('Coefficients:', MyLinearRegression.coef_) #this tells the coefficients for each input, aka the impact they have

#shows how accurate your current regression model is
CurrentModelAccuracy = MyLinearRegression.score(x_test, y_test)
print('Current Model Accuracy:', CurrentModelAccuracy)
PickleModelAccuracy = picklemodelG3.score(x_test, y_test)
print("Pickle Model Accuracy:", PickleModelAccuracy)

