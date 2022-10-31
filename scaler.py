import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from data_cleaner import full_cleaner

start_time = time.time()

dataframe = full_cleaner()
target_variable = 'G3'
print('data:\n', dataframe)
features = dataframe.drop([target_variable], axis=1)
print('features:\n', features)
X = np.array(features)
y = np.array(dataframe[target_variable])
runtimes = 1000

# target features do not need to be scaled generally
def standardizer():
    total_accuracy = 0
    scaler = sklearn.preprocessing.StandardScaler()
    standardized_array = scaler.fit_transform(X)
    standardized_df = pd.DataFrame(standardized_array, columns= features.columns) # problem line
    standardized_X = np.array(standardized_df)

    for i in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            standardized_X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    standardizer_average_accuracy = total_accuracy / runtimes
    print('Standardizer Average Accuracy:', standardizer_average_accuracy)
    return standardizer_average_accuracy, standardized_df


def normalizer():
    total_accuracy = 0
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(X)
    normalized_df = pd.DataFrame(normalized_array, columns=features.columns)
    normalized_X = np.array(normalized_df)

    for i in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            normalized_X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    normalizer_average_accuracy = total_accuracy/runtimes
    print('Normalizer Average Accuracy:', normalizer_average_accuracy)
    return normalizer_average_accuracy, normalized_df

def raw():
    total_accuracy = 0
    for i in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    raw_average_accuracy = total_accuracy / runtimes
    print('Raw Average Accuracy:', raw_average_accuracy)
    return raw_average_accuracy

def main_scaler():
    raw_accuracy = raw()
    standardized_accuracy, standardized_df = standardizer()
    normalized_accuracy, normalized_df = normalizer()
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
        scaled_df = dataframe
    print('Winner is', winner)
    time_elapsed = time.time() - start_time
    print('Time Elapsed:', time_elapsed, 'seconds')
    scaled_df = pd.concat([dataframe[target_variable], scaled_df], axis=1)
    scaled_df.to_csv('scaled_dataframe.csv', index=False, encoding='utf-8')
    return scaled_df

def main_non_printing_scaler():
    raw_accuracy = raw()
    standardized_accuracy, standardized_df = standardizer()
    normalized_accuracy, normalized_df = normalizer()

    if normalized_accuracy > standardized_accuracy and normalized_accuracy > raw_accuracy:
        winner = 'normalizer'
        scaled_df = normalized_df
    if standardized_accuracy > normalized_accuracy and standardized_accuracy > raw_accuracy:
        winner = 'standardizer'
        scaled_df = standardized_df
    if raw_accuracy > normalized_accuracy and raw_accuracy > standardized_accuracy:
        winner = 'raw'
        scaled_df = dataframe
    time_elapsed = time.time() - start_time
    scaled_df = pd.concat([dataframe[target_variable], scaled_df], axis=1)
    scaled_df.to_csv('scaled_dataframe.csv', index=False, encoding='utf-8')
    return scaled_df


if __name__ == '__main__':
    main_scaler()

