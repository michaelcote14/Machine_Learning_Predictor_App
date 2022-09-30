import numpy as np
import pandas_tutorital as pd

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))
UploadFile = "Feature Optimizer Data.csv"
Dataframe = pd.read_csv(UploadFile, sep=',')
PickedFeatures = ['Fedu', 'Medu', 'age', 'G3']
DataPicks = Dataframe[PickedFeatures]
TargetVariable = "G3"
X = np.array(DataPicks.drop([TargetVariable], axis=1))
y = np.array(DataPicks[TargetVariable])


x_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #add in randomstate= a # to stop randomly changing your arrays

MyLinearRegression = linear_model.LinearRegression().fit(x_train, y_train)