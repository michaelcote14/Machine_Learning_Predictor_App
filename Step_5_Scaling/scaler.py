import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Extras.functions import time_formatter
import time

def scaler_time_predictor(runtimes, fully_cleaned_df):
    scaler_predicted_time = runtimes * len(fully_cleaned_df) * len(fully_cleaned_df) * 3 * 0.00000062672
    scaler_predicted_time = time_formatter(scaler_predicted_time)
    return scaler_predicted_time


def standardizer(runtimes, X, y, unscaled_predictor_array, FEATURES, scaler_progressbar, master_frame):
    total_accuracy = 0
    scaler = sklearn.preprocessing.StandardScaler()
    standardized_array = scaler.fit_transform(X)
    standardized_predictor_array = scaler.transform(unscaled_predictor_array)
    standardized_df = pd.DataFrame(standardized_array, columns= FEATURES.columns)
    standardized_X = np.array(standardized_df)

    for i in range(runtimes):
        scaler_progressbar['value'] += 100/runtimes/3
        master_frame.update_idletasks()



        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            standardized_X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    standardizer_average_accuracy = total_accuracy / runtimes
    return standardizer_average_accuracy, standardized_df, standardized_predictor_array


def normalizer(runtimes, X, y, unscaled_predictor_array, FEATURES, scaler_progressbar, master_frame):
    # brings the scaler in and applies it to the original array as well as the predicted array
    total_accuracy = 0
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(X)
    normalized_predictor_array = scaler.transform(unscaled_predictor_array)

    normalized_df = pd.DataFrame(normalized_array, columns=FEATURES.columns)
    normalized_X = np.array(normalized_df)

    for i in range(runtimes):
        scaler_progressbar['value'] += 100/runtimes/3
        master_frame.update_idletasks()



        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            normalized_X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    normalizer_average_accuracy = total_accuracy/runtimes
    return normalizer_average_accuracy, normalized_df, normalized_predictor_array

def raw(runtimes, X, y, scaler_progressbar, master_frame):
    total_accuracy = 0
    for i in range(runtimes):
        scaler_progressbar['value'] += 100/runtimes/3
        master_frame.update_idletasks()


        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    raw_average_accuracy = total_accuracy / runtimes
    return raw_average_accuracy

def main_scaler(runtimes, fully_cleaned_df, target_variable, data_we_know_dict, scaler_progressbar, master_frame, split_dataframe_question):
    start_time = time.time()

    FEATURES = fully_cleaned_df.drop([target_variable], axis=1)
    X = np.array(FEATURES)
    y = np.array(fully_cleaned_df[target_variable])

    if split_dataframe_question == 'no':
        unscaled_predictor_array = get_predictor_array(data_we_know_dict, fully_cleaned_df, target_variable)
    else:
        unscaled_predictor_array = get_predictor_array({}, fully_cleaned_df, target_variable)

    raw_accuracy = raw(runtimes, X, y, scaler_progressbar, master_frame)
    standardized_accuracy, standardized_df, standardized_predictor_array = standardizer(runtimes, X, y, unscaled_predictor_array, FEATURES, scaler_progressbar, master_frame)
    normalized_accuracy, normalized_df, normalized_predictor_array = normalizer(runtimes, X, y, unscaled_predictor_array, FEATURES, scaler_progressbar, master_frame)


    if normalized_accuracy > standardized_accuracy and normalized_accuracy > raw_accuracy:
        winner = 'normalizer'
        scaled_df = normalized_df
        scaled_predictor_array = normalized_predictor_array
        scaled_df = pd.concat([fully_cleaned_df[target_variable], scaled_df], axis=1)

    if standardized_accuracy > normalized_accuracy and standardized_accuracy > raw_accuracy:
        winner = 'standardizer'
        scaled_df = standardized_df
        scaled_predictor_array = standardized_predictor_array
        scaled_df = pd.concat([fully_cleaned_df[target_variable], scaled_df], axis=1)

    if raw_accuracy > normalized_accuracy and raw_accuracy > standardized_accuracy:
        winner = 'raw'
        scaled_df = fully_cleaned_df
        scaled_predictor_array = unscaled_predictor_array
    pd.options.display.width = 500
    pd.set_option('display.max_columns', 80)

    scaled_data_we_know_df = pd.DataFrame(scaled_predictor_array, columns=FEATURES.columns)

    elapsed_time = time.time() - start_time

    return scaled_df, scaled_data_we_know_df

def get_predictor_array(data_we_know_dict, fully_cleaned_df, target_variable):
    data_we_have_dataframe = pd.DataFrame.from_dict(data_we_know_dict)
    # combines the two dataframes and gives null values the mean
    new_dataframe = pd.concat([data_we_have_dataframe, fully_cleaned_df])
    new_dataframe.fillna(fully_cleaned_df.mean())
    # separates the two dataframes again
    new_dataframe = new_dataframe.iloc[0]
    new_dataframe = pd.DataFrame(new_dataframe)
    new_dataframe = new_dataframe.T
    new_dataframe = new_dataframe.drop([target_variable], axis=1)
    unscaled_predictor_array = new_dataframe.fillna(fully_cleaned_df.mean())
    unscaled_predictor_array = np.array(unscaled_predictor_array)
    return unscaled_predictor_array # not in order


