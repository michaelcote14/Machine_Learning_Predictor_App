import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle
import time
import multiple_hot_encoder
from sklearn.model_selection import cross_val_score


start_time=time.time()

RunEvalution = 'Yes'
dataframe = multiple_hot_encoder.multiple_encoder()
data=dataframe[['G3','G2','G1','age','goout','romantic_yes','traveltime','paid_yes','internet_yes','studytime']]

target_variable = 'G3'
X = np.array(data.drop([target_variable],axis=1))
y = np.array(data[target_variable])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)

def predictor():
    # this makes it to where you can predict only off of the data you have, the rest is the mean of the data
    data_we_have = {'age': 16, 'G2': 10, 'goout': 50}
    key_lst = list(data_we_have.keys())
    value_lst = list(data_we_have.values())
    #createalistofindexestoget
    index_lst = []
    for index in range(len(key_lst)):
        indexes_to_get = data.columns.get_loc(key_lst[index])
        index_lst.append(indexes_to_get)
    print('index list', index_lst)
    #gets the means for all columns
    predictor_lst = []
    for index, column in enumerate(data.columns):
        #makes a list with all the means
        predictor_lst.append(data[column].mean())
    print(predictor_lst)
    #replace the age mean with 16
    for count, index in enumerate(index_lst):
        print('index', index)
        print('count:', count)
        for value in value_lst:
            print('value', value)
            predictor_lst[index] = value_lst[count] #this replaces both columns with 10
    del predictor_lst[0]
    print('predictor list:', predictor_lst)
    print(data.columns[1:100].tolist())

    CurrentModelsPredictions = MyLinearRegression.predict(X_test)


    CurrentModelsInputPrediction = MyLinearRegression.predict([predictor_lst]) #problem line
    current_cross_val_score = cross_val_score(MyLinearRegression, X, y, cv=10).mean()
    current_normal_score = MyLinearRegression.score(X_test, y_test)
    print(CurrentModelsInputPrediction)


predictor()
