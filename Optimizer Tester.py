import pandas_tutorital as pd #this is to read in data sheets
import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot #this allows you to make graphs
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
from matplotlib import style #this changes the style of your plot's grid

UploadFile = "student-mat(Numerical Only).csv"
Dataframe = pd.read_csv(UploadFile, sep=',')
print("Dataframe Columns:", Dataframe.columns)
DataColumns = Dataframe.columns
DataFeatures = DataColumns.drop("G3")
print("DataFeatures:", DataFeatures)



#what we need: Dataframe[["Medu", "Fedu", "G1", "G2", "studytime", "famrel", "freetime", "traveltime", "failures", "health", "Walc", "Dalc", "G3"]]
TargetVariable = "G3"
BestScore = 0
CombinationCount = 0
BestFeatures = []
for i in DataFeatures:
    for j in DataFeatures:
        Features = [i, j]
        print("Features:", Features)
        DataPicks = Dataframe[[i, j]]
        X = np.array(DataPicks) #this is not an array yet, but its supposed to be
        y = np.array(Dataframe[TargetVariable])


        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=0)  # add in randomstate= a # to stop randomly changing your arrays

        MyLinearRegression = linear_model.LinearRegression()
        MyLinearRegression.fit(x_train, y_train)  # creates the regression line
        MyLinearRegressionScore = MyLinearRegression.score(x_test, y_test)
        CombinationCount = CombinationCount+1
        if MyLinearRegressionScore > BestScore:
           BestScore = MyLinearRegressionScore
           BestFeatures = Features

print("Best Score:", BestScore)
print("Best Features:", BestFeatures)
print("Combination Count:", CombinationCount)

