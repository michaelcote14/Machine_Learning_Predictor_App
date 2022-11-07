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
TARGET_VARIABLE = 'G3'

FEATURES = dataframe.drop([TARGET_VARIABLE], axis=1)
X = np.array(FEATURES)
y = np.array(dataframe[TARGET_VARIABLE])
RUNTIMES = 100

def get_predictor_array(predictor_data_dict):
    data_we_have_dataframe = pd.DataFrame.from_dict(predictor_data_dict)
    # combines the two dataframes and gives null values the mean
    new_dataframe = pd.concat([data_we_have_dataframe, dataframe])
    new_dataframe.fillna(dataframe.mean())
    # separates the two dataframes again
    new_dataframe = new_dataframe.iloc[0]
    new_dataframe = pd.DataFrame(new_dataframe)
    new_dataframe = new_dataframe.T
    new_dataframe = new_dataframe.drop([TARGET_VARIABLE], axis=1)
    unscaled_predictor_array = new_dataframe.fillna(dataframe.mean())
    unscaled_predictor_array = np.array(unscaled_predictor_array)
    #ToDo put predictor data in correct column locations
    #steps: get original dataframe mean values in a dataframe
    return unscaled_predictor_array # not in order


def predictor_data_scaler(predictor_data_dict):
    unscaled_predictor_array = get_predictor_array(predictor_data_dict)
    scaled_df, scaled_predictor_array = main_scaler(unscaled_predictor_array)
    scaled_predictor_df = pd.DataFrame(scaled_predictor_array, columns=FEATURES.columns) #ToDo this is wrong
    # data = scaled_df[['G3', 'G2', 'G1', 'age', 'goout', 'romantic_yes', 'traveltime', 'paid_yes', 'internet_yes', 'studytime']]
    # this creates the dataframe with scaled data from input data only
    key_lst = list(predictor_data_dict.keys())
    value_lst = list(predictor_data_dict.values())
    scaled_predictor_df = scaled_predictor_df.loc[:, key_lst]
    scaled_predictor_df.to_csv('scaled_predictor_df.csv', index=False, encoding='utf-8')
    return scaled_predictor_array, scaled_predictor_df


# target FEATURES do not need to be scaled generally
def standardizer(unscaled_predictor_array):
    total_accuracy = 0
    scaler = sklearn.preprocessing.StandardScaler()
    standardized_array = scaler.fit_transform(X)
    standardized_predictor_array = scaler.transform(unscaled_predictor_array)
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
    return standardizer_average_accuracy, standardized_df, standardized_predictor_array


def normalizer(unscaled_predictor_array):
    # brings the scaler in and applies it to the original array as well as the predicted array
    total_accuracy = 0
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(X)
    normalized_predictor_array = scaler.transform(unscaled_predictor_array)

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
    return normalizer_average_accuracy, normalized_df, normalized_predictor_array

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

def main_scaler(unscaled_predictor_array):
    raw_accuracy = raw()
    standardized_accuracy, standardized_df, standardized_predictor_array = standardizer(unscaled_predictor_array)
    normalized_accuracy, normalized_df, normalized_predictor_array = normalizer(unscaled_predictor_array)
    print('\nNormalizer Average Accuracy: ', normalized_accuracy)
    print('Standardizer Average Accuracy', standardized_accuracy)
    print('Raw Average Accuracy         ', raw_accuracy)


    if normalized_accuracy > standardized_accuracy and normalized_accuracy > raw_accuracy:
        winner = 'normalizer'
        scaled_df = normalized_df
        scaled_predictor_array = normalized_predictor_array
    if standardized_accuracy > normalized_accuracy and standardized_accuracy > raw_accuracy:
        winner = 'standardizer'
        scaled_df = standardized_df
        scaled_predictor_array = standardized_predictor_array
    if raw_accuracy > normalized_accuracy and raw_accuracy > standardized_accuracy:
        winner = 'raw'
        scaled_df = dataframe
        scaled_predictor_array = unscaled_predictor_array
    print('Winner is', winner)
    time_elapsed = time.time() - start_time
    print('Time Elapsed:', time_elapsed, 'seconds')
    scaled_df = pd.concat([dataframe[TARGET_VARIABLE], scaled_df], axis=1)
    scaled_df.to_csv('scaled_dataframe.csv', index=False, encoding='utf-8')
    return scaled_df, scaled_predictor_array

def main_scaler_non_printing(unscaled_predictor_array):
    raw_accuracy = raw()
    standardized_accuracy, standardized_df, standardized_predictor_array = standardizer(unscaled_predictor_array)
    normalized_accuracy, normalized_df, normalized_predictor_array = normalizer(unscaled_predictor_array)

    if normalized_accuracy > standardized_accuracy and normalized_accuracy > raw_accuracy:
        winner = 'normalizer'
        scaled_df = normalized_df
        scaled_predictor_array = normalized_predictor_array
    if standardized_accuracy > normalized_accuracy and standardized_accuracy > raw_accuracy:
        winner = 'standardizer'
        scaled_df = standardized_df
        scaled_predictor_array = standardized_predictor_array
    if raw_accuracy > normalized_accuracy and raw_accuracy > standardized_accuracy:
        winner = 'raw'
        scaled_df = dataframe
        scaled_predictor_array = unscaled_predictor_array
    time_elapsed = time.time() - start_time
    scaled_df = pd.concat([dataframe[TARGET_VARIABLE], scaled_df], axis=1)
    scaled_df.to_csv('scaled_dataframe.csv', index=False, encoding='utf-8')
    return scaled_df, scaled_predictor_array




if __name__ == '__main__':
    unscaled_predictor_array = get_predictor_array(predictor_data_dict = {'age': [16.0], 'G2': [10.0], 'goout': [3], 'internet_yes': [1]})
    print(unscaled_predictor_array)

    # # ToDo transfer the above dataframe to the predictor
    # ToDo what we need: an array with all the means and the scaled data
    # ToDo why are some mean values in the array 0?

