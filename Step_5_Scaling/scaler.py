import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Extras.functions import time_formatter
import time


class Scaler:
    def scaler_time_predictor(self, runtimes, fully_cleaned_df):
        scaler_predicted_time = runtimes * len(fully_cleaned_df) * len(fully_cleaned_df) * 3 * 0.00000062672
        scaler_predicted_time = time_formatter(scaler_predicted_time)
        return scaler_predicted_time

    def standardizer(self, runtimes, X, y, dataframe, scaler_progressbar, master_frame):
        total_accuracy = 0
        self.standard_scaler = sklearn.preprocessing.StandardScaler()
        standardized_array = self.standard_scaler.fit_transform(X)
        standardized_df = pd.DataFrame(standardized_array, columns=dataframe.columns)
        standardized_X = np.array(standardized_df)

        for i in range(runtimes):
            scaler_progressbar['value'] += 100 / runtimes / 3
            master_frame.update_idletasks()

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                standardized_X, y, test_size=0.2)
            linear = linear_model.LinearRegression()

            linear.fit(X_train, y_train)
            accuracy = linear.score(X_test, y_test)
            total_accuracy += accuracy
        standardizer_average_accuracy = total_accuracy / runtimes
        return standardizer_average_accuracy, standardized_df

    def normalizer(self, runtimes, X, y, dataframe, scaler_progressbar, master_frame):
        # brings the scaler in and applies it to the original array as well as the predicted array
        total_accuracy = 0
        self.min_max_scaler = MinMaxScaler()
        normalized_array = self.min_max_scaler.fit_transform(X)

        normalized_df = pd.DataFrame(normalized_array, columns=dataframe.columns)
        normalized_X = np.array(normalized_df)

        for i in range(runtimes):
            scaler_progressbar['value'] += 100 / runtimes / 3
            master_frame.update_idletasks()

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                normalized_X, y, test_size=0.2)
            linear = linear_model.LinearRegression()

            linear.fit(X_train, y_train)
            accuracy = linear.score(X_test, y_test)
            total_accuracy += accuracy
        normalizer_average_accuracy = total_accuracy / runtimes
        return normalizer_average_accuracy, normalized_df

    def raw(self, runtimes, X, y, scaler_progressbar, master_frame):
        total_accuracy = 0
        for i in range(runtimes):
            scaler_progressbar['value'] += 100 / runtimes / 3
            master_frame.update_idletasks()

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                X, y, test_size=0.2)
            linear = linear_model.LinearRegression()

            linear.fit(X_train, y_train)
            accuracy = linear.score(X_test, y_test)
            total_accuracy += accuracy
        raw_average_accuracy = (total_accuracy / runtimes) + 100
        return raw_average_accuracy

    def main_scaler(self, runtimes, target_variable, scaler_progressbar, master_frame, fully_cleaned_df,
                    fully_cleaned_df2):
        start_time = time.time()

        dataframe = fully_cleaned_df.drop([target_variable], axis=1)

        X = np.array(dataframe)
        y = np.array(fully_cleaned_df[target_variable])
        X2 = np.array(fully_cleaned_df2)

        raw_accuracy = Scaler.raw(self, runtimes, X, y, scaler_progressbar, master_frame)
        standardized_accuracy, standardized_df = Scaler.standardizer(self, runtimes, X, y, dataframe,
                                                                     scaler_progressbar, master_frame)
        normalized_accuracy, normalized_df = Scaler.normalizer(self, runtimes, X, y, dataframe, scaler_progressbar,
                                                               master_frame)

        if normalized_accuracy > standardized_accuracy and normalized_accuracy > raw_accuracy:
            winner = 'normalizer'
            scaled_df = normalized_df
            scaled_df = pd.concat([fully_cleaned_df[target_variable], scaled_df], axis=1)

        if standardized_accuracy > normalized_accuracy and standardized_accuracy > raw_accuracy:
            winner = 'standardizer'
            scaled_df = standardized_df
            scaled_df = pd.concat([fully_cleaned_df[target_variable], scaled_df], axis=1)

        if raw_accuracy > normalized_accuracy and raw_accuracy > standardized_accuracy:
            winner = 'raw'
            scaled_df = fully_cleaned_df

        if winner == 'raw':
            scaled_df2 = fully_cleaned_df2

        if winner == 'standardizer':
            standardized_array2 = self.standard_scaler.transform(X2)
            scaled_df2 = pd.DataFrame(standardized_array2, columns=dataframe2.columns)

        if winner == 'normalizer':
            normalized_array2 = self.min_max_scaler.transform(X2)
            scaled_df2 = pd.DataFrame(normalized_array2, columns=dataframe2.columns)

        pd.options.display.width = 500
        pd.set_option('display.max_columns', 80)

        elapsed_time = time.time() - start_time

        print('scaled_df:\n', scaled_df)
        print('scaled df2:\n', scaled_df2)
        return scaled_df, scaled_df2

    def get_predictor_array(self, data_we_know_dict, fully_cleaned_df, target_variable):
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
        return unscaled_predictor_array  # not in order
