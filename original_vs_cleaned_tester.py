import sklearn
from sklearn import linear_model
import numpy as np
import data_cleaner
import multiple_hot_encoder
import pandas as pd
import scaler
from multiple_hot_encoder import multiple_encoder

def encoder_tester(runtimes):
    df = pd.read_csv('Data/student-mat.csv')
    data_cleaner.unnamed_column_dropper(df)
    for column in df:
        if df[column].dtypes == 'object':
            df.drop(column, axis=1, inplace=True)

    encoded_df = multiple_hot_encoder.multiple_encoder()
    encoded_df = data_cleaner.unnamed_column_dropper(encoded_df)

    target_variable = 'G3'
    X = np.array(df.drop([target_variable], axis=1))
    y = np.array(df[target_variable])

    original_total_score, encoded_total_score = 0, 0
    for loops in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        original_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)
        original_score = original_linear_regression.score(X_test, y_test)
        original_total_score = original_total_score + original_score
        original_average_score = original_total_score/runtimes


        X = np.array(encoded_df.drop([target_variable], axis=1))
        y = np.array(encoded_df[target_variable])
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        encoded_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)
        encoded_score = encoded_linear_regression.score(X_test, y_test)
        encoded_total_score = encoded_total_score + encoded_score
        encoded_average_score = encoded_total_score/runtimes
    print('\nEncoded Average Score:', encoded_average_score)
    print('Original Average Score:', original_average_score)
    print('Encoded - Original Score:', encoded_average_score - original_average_score)

def outlier_tester(runtimes):
    df = multiple_encoder()
    data_cleaner.unnamed_column_dropper(df)
    outlier_free_df = data_cleaner.outlier_cleaner() #still has unnamed in it
    data_cleaner.unnamed_column_dropper(outlier_free_df)
    target_variable = 'G3'
    X = np.array(df.drop([target_variable], axis=1))
    y = np.array(df[target_variable])

    original_total_score, outlier_total_score = 0, 0
    for loops in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        original_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)
        original_score = original_linear_regression.score(X_test, y_test)
        original_total_score = original_total_score + original_score
        original_average_score = original_total_score / runtimes

        X = np.array(outlier_free_df.drop([target_variable], axis=1))
        y = np.array(outlier_free_df[target_variable])
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        outlier_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)
        outlier_score = outlier_linear_regression.score(X_test, y_test)
        outlier_total_score = outlier_total_score + outlier_score
        outlier_average_score = outlier_total_score / runtimes

    print('\nOutlier Average Score:', outlier_average_score)
    print('Original Average Score:', original_average_score)
    print('Outlier - Original Score:', outlier_average_score - original_average_score)

def scaler_tester(runtimes):
    df = data_cleaner.full_cleaner()
    scaled_df = scaler.main_scaler()
    target_variable = 'G3'
    X = np.array(df.drop([target_variable], axis=1))
    y = np.array(df[target_variable])

    original_total_score, scaled_total_score = 0, 0
    for loops in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        original_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)
        original_score = original_linear_regression.score(X_test, y_test)
        original_total_score = original_total_score + original_score
        original_average_score = original_total_score / runtimes

        X = np.array(scaled_df.drop([target_variable], axis=1))
        y = np.array(scaled_df[target_variable])
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        scaled_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)
        scaled_score = scaled_linear_regression.score(X_test, y_test)
        scaled_total_score = scaled_total_score + scaled_score
        scaled_average_score = scaled_total_score / runtimes

    print('\nScaled Average Score:', scaled_average_score)
    print('Scaled Average Score:', original_average_score)
    print('Scaled - Original Score:', scaled_average_score - original_average_score)

def total_tester(runtimes):
    df = pd.read_csv('Data/student-mat.csv')
    data_cleaner.unnamed_column_dropper(df)
    for column in df:
        if df[column].dtypes == 'object':
            df.drop(column, axis=1, inplace=True)

    last_df = scaler.main_scaler()

    target_variable = 'G3'
    X = np.array(df.drop([target_variable], axis=1))
    y = np.array(df[target_variable])

    original_total_score, last_total_score = 0, 0
    for loops in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        original_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)
        original_score = original_linear_regression.score(X_test, y_test)
        original_total_score = original_total_score + original_score
        original_average_score = original_total_score/runtimes

        X = np.array(last_df.drop([target_variable], axis=1))
        y = np.array(last_df[target_variable])
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        last_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)
        last_score = last_linear_regression.score(X_test, y_test)
        last_total_score = last_total_score + last_score
        last_average_score = last_total_score / runtimes

    print('\033[34m' + '\nLast Average Score:', last_average_score)
    print('Original Average Score:', original_average_score)
    print('Last - Original Score:', last_average_score - original_average_score)

if __name__ == '__main__':
    # encoder_tester(100)
    # outlier_tester(100)
    # scaler_tester(100)
    total_tester(100)
