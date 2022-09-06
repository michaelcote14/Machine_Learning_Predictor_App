import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

Dataframe = pd.read_csv("student-mat.csv", sep=';')

DataPicks = Dataframe[["G1", "G2", "G3", "studytime", "failures", "absences"]]

PredictorVariableG3 = "G3"

X = np.array(DataPicks.drop([PredictorVariableG3], axis=1))
y = np.array(DataPicks[PredictorVariableG3])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

MyLinearRegression = linear_model.LinearRegression()

MyLinearRegression.fit(x_train, y_train)
MyLinearRegressionAccuracy = MyLinearRegression.score(x_test, y_test)
print(MyLinearRegressionAccuracy)

MyModelsPredictions = MyLinearRegression.predict(x_test)
for x in range(len(MyModelsPredictions)):
    print(MyModelsPredictions[x], x_test[x], y_test[x])