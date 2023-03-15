import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import rfpimp
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
import time
from Extras import functions
import pandas as pd
import random
from Step_1_Data_Cleaning import data_cleaner
import math


dataframe = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_Starting_Dataframes/real_estate_data/train(numerical).csv")
target_variable = 'SalePrice'

# Shorten the dataframe to test curse of dimensionality
dataframe = dataframe.iloc[0:50, :]
print(dataframe)

# Clean the dataframe
dataframe, columns_removed, rows_removed = data_cleaner.full_cleaner(dataframe, target_variable)

def feature_importance_finder(runtimes, dataframe, target_variable, feature_length_wanted):
    ######################################## Data preparation #########################################

    pd.set_option('display.max_columns', 85)
    all_features = dataframe.columns.tolist()

    ######################################## Train/test split #########################################

    train_df, test_df = train_test_split(dataframe, test_size=0.20, random_state=0)
    train_df = train_df[all_features]
    test_df = test_df[all_features]

    X_train, y_train = train_df.drop(target_variable, axis=1), train_df[target_variable]
    X_test, y_test = test_df.drop(target_variable, axis=1), test_df[target_variable]

    # ################################################ Train #############################################
    rf = RandomForestRegressor(n_estimators=int(runtimes), n_jobs=-1)
    rf.fit(X_train, y_train)

    # ############################### Permutation feature importance #####################################
    importance_series = rfpimp.importances(rf, X_test, y_test)

    # Turn the above series into a dictionary
    importance_series_index_list = importance_series.index.tolist()
    importance_dictionary = {}

    # Sort the dictionary by the values
    loop_number = 0
    for i in importance_series_index_list:
        importance_dictionary[i] = importance_series['Importance'][loop_number]
        loop_number = loop_number + 1
    sorted_dict = dict(sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True))

    # Now put the above sorted dictionary into a list
    most_important_features = []
    for n in range(int(feature_length_wanted)):
        new_corr_list = list(sorted_dict)
        most_important_features.append(new_corr_list[n])

    importances_of_all_features = list(sorted_dict.values())
    print('Importances of All Features:', importances_of_all_features)
    print('    Most Important Features:', most_important_features)

    # Add the target variable to the list of most important features
    most_important_features.insert(0, target_variable)

    print('\nPlotting feature importance...')
    importance_plotter(most_important_features, importances_of_all_features, feature_length_wanted)

    # ToDo set it up to where it grabs as many features as it can without hitting the curse of dimensionality

    return most_important_features

def importance_plotter(most_important_features, most_important_values, feature_length_wanted):
    fig, ax = plt.subplots(figsize=(20, 18))
    ax.barh(most_important_features[1:], most_important_values[:int(feature_length_wanted)], height=0.8, facecolor='grey', alpha=0.8, edgecolor='k')
    ax.set_xlabel('Importance score')
    ax.set_title('Permutation feature importance')
    ax.text(0.8, 0.15, 'aegis4048.github.io', fontsize=12, ha='center', va='center',
            transform=ax.transAxes, color='grey', alpha=0.5)
    plt.gca().invert_yaxis()
    fig.tight_layout()
    plt.show()

def feature_accuracy_measurer(runtimes, dataframe, target_variable):
    X = np.array(dataframe.drop([target_variable], axis=1))
    y = np.array(dataframe[target_variable])

    total_accuracy = 0
    for _ in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        random_model_regression_line = linear_model.LinearRegression()
        random_model_regression_line.fit(X_train, y_train)

        accuracy = random_model_regression_line.score(X_test, y_test)
        total_accuracy = total_accuracy + accuracy

    average_accuracy = total_accuracy / runtimes

    return average_accuracy


if __name__ == '__main__':
    most_important_features = feature_importance_finder(1000, dataframe, target_variable, 5)

    important_features_accuracy = feature_accuracy_measurer(1000, dataframe[most_important_features], target_variable)
    all_features_accuracy = feature_accuracy_measurer(1000, dataframe, target_variable)

    print('\nImportant Features Accuracy:', important_features_accuracy)
    print('All Features Accuracy:     ', all_features_accuracy)
    print('Difference:                 ', important_features_accuracy - all_features_accuracy)

    print('\nlen of dataframe:', len(dataframe))
    print('len of dataframe.columns:', len(dataframe.columns))
    print('best amount of features:', math.floor(len(dataframe)/10))
