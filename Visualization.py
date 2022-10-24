import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
import time
import functions
import multiple_hot_encoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


original_df = multiple_hot_encoder.multiple_encoder()
df = original_df[['G3', 'G2', 'G1', 'age', 'goout', 'romantic_yes', 'traveltime', 'paid_yes', 'internet_yes', 'studytime']]

target_variable = 'G3'
X = np.array(df.drop([target_variable], axis=1))
y = np.array(df[target_variable])

'''sns.lmplot(x = 'traveltime', y = 'G3', data=df)
plt.show()'''


# how to plot your predictions
PredictorInputData = [4, 4, 4, 4, 4, 4, 4, 4, 4]
PicklePredictorInputData = [4, 4, 4, 4, 4, 4, 4, 4, 4]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
y_pred = MyLinearRegression.predict(X_test)
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values
print('y:', y.shape)
print('X', X.shape)
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('x_test:', X_test.shape)
print('y_test', y_test.shape)
y_pred = MyLinearRegression.predict(X_test)
print('y pred:', y_pred)

plt.scatter(X_train, y_train, color='green')
plt.plot(X_test, y_pred, color='red')
plt.title('IDK')
plt.xlabel('x label')
plt.ylabel('G3')
plt.show()

