import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from Step_3_Multiple_Encoding.multiple_hot_encoder import multiple_encoded_df
from Step_1_Visualizing.visualization import target_variable

start_time = time.time()
FEATURES = multiple_encoded_df.drop([target_variable], axis=1)
X = np.array(FEATURES)
y = np.array(multiple_encoded_df[target_variable])
RUNTIMES = 100

def standardizer(dataframe):
    total_accuracy = 0
    scaler = sklearn.preprocessing.StandardScaler()
    standardized_array = scaler.fit_transform(X)
    standardized_df = pd.DataFrame(standardized_array, columns= FEATURES.columns) # problem line
    standardized_X = np.array(standardized_df)

    for i in range(RUNTIMES):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            standardized_X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    standardizer_average_accuracy = total_accuracy / RUNTIMES
    return standardizer_average_accuracy, standardized_df


def normalizer(dataframe):
    # brings the scaler in and applies it to the original array as well as the predicted array
    total_accuracy = 0
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(X)

    normalized_df = pd.DataFrame(normalized_array, columns=FEATURES.columns)
    normalized_X = np.array(normalized_df)

    for i in range(RUNTIMES):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            normalized_X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    normalizer_average_accuracy = total_accuracy/RUNTIMES
    return normalizer_average_accuracy, normalized_df

def raw():
    total_accuracy = 0
    for i in range(RUNTIMES):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    raw_average_accuracy = total_accuracy / RUNTIMES
    return raw_average_accuracy

def main_scaler(dataframe):
    raw_accuracy = raw()
    standardized_accuracy, standardized_df = standardizer(dataframe)
    normalized_accuracy, normalized_df = normalizer(dataframe)
    print('\nNormalizer Average Accuracy: ', normalized_accuracy)
    print('Standardizer Average Accuracy', standardized_accuracy)
    print('Raw Average Accuracy         ', raw_accuracy)


    if normalized_accuracy > standardized_accuracy and normalized_accuracy > raw_accuracy:
        winner = 'normalizer'
        scaled_df = normalized_df
    if standardized_accuracy > normalized_accuracy and standardized_accuracy > raw_accuracy:
        winner = 'standardizer'
        scaled_df = standardized_df
    if raw_accuracy > normalized_accuracy and raw_accuracy > standardized_accuracy:
        winner = 'raw'
        scaled_df = multiple_encoded_df
    print('Winner is', winner)
    time_elapsed = time.time() - start_time
    scaled_df = pd.concat([multiple_encoded_df[target_variable], scaled_df], axis=1)
    scaled_df.to_csv('scaled_dataframe.csv', index=False, encoding='utf-8')
    return scaled_df

def main_scaler_non_printing(dataframe):
    raw_accuracy = raw()
    standardized_accuracy, standardized_df = standardizer(dataframe)
    normalized_accuracy, normalized_df = normalizer(dataframe)
    print('\nNormalizer Average Accuracy: ', normalized_accuracy)
    print('Standardizer Average Accuracy', standardized_accuracy)
    print('Raw Average Accuracy         ', raw_accuracy)


    if normalized_accuracy > standardized_accuracy and normalized_accuracy > raw_accuracy:
        winner = 'normalizer'
        scaled_df = normalized_df
    if standardized_accuracy > normalized_accuracy and standardized_accuracy > raw_accuracy:
        winner = 'standardizer'
        scaled_df = standardized_df
    if raw_accuracy > normalized_accuracy and raw_accuracy > standardized_accuracy:
        winner = 'raw'
        scaled_df = multiple_encoded_df
    print('Winner is', winner)
    time_elapsed = time.time() - start_time
    scaled_df = pd.concat([multiple_encoded_df[target_variable], scaled_df], axis=1)
    scaled_df.to_csv('scaled_dataframe.csv', index=False, encoding='utf-8')
    return scaled_df


scaled_df = main_scaler_non_printing(multiple_encoded_df)
if __name__ == '__main__':
    print('Scaled Df:\n', scaled_df)
    pass

