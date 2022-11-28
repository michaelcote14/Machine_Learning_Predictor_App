import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import time
from Extras import functions
import pandas as pd
from Step_1_Visualizing.visualization import target_variable
import random

# ToDo put in a time predictor and percentage tracker

scaled_df = pd.read_csv('../Step_5_Scaling/scaled_dataframe.csv')
with open('../Data/pickled_most_recent_important_features', 'rb') as f:
    most_important_features = pickle.load(f)[:]
print('Length of Features:', len(most_important_features))
small_loops = 10

def most_important_features_measurer(runtimes):
    importance_model_total_accuracy = 0
    df = scaled_df[most_important_features]

    for _ in range(runtimes):
        X = np.array(df.drop([target_variable], axis=1))
        y = np.array(df[target_variable])

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        importance_model_regression_line = linear_model.LinearRegression()
        importance_model_regression_line.fit(X_train, y_train)
        importance_model_accuracy = importance_model_regression_line.score(X_test, y_test)
        importance_model_total_accuracy = importance_model_total_accuracy + importance_model_accuracy
    importance_model_average_accuracy = importance_model_total_accuracy / small_loops
    print('\nImportance Model Average Accuracy:', importance_model_average_accuracy)

def regular_features_measurer(runtimes):

    columns_list = scaled_df.columns.tolist()
    print('Columns List:', columns_list)
    print('Length of Column List:', len(columns_list))

    best_average_accuracy = 0
    best_features = []
    for _ in range(runtimes):
        # makes a random dataframe
        random_int = random.randint(0, len(columns_list))
        random_list = random.choices(columns_list, k=random_int)
        print('\nRandom List:', random_list)
        random_df = scaled_df[random_list]
        # measures the accuracy of this dataframe
        X = np.array(scaled_df.drop([target_variable], axis=1))
        y = np.array(scaled_df[target_variable])

        random_model_total_accuracy = 0
        for _ in range(small_loops):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            random_model_regression_line = linear_model.LinearRegression()
            random_model_regression_line.fit(X_train, y_train)
            random_model_accuracy = random_model_regression_line.score(X_test, y_test)
            print('Accuracy:', random_model_accuracy)
            random_model_total_accuracy = random_model_total_accuracy + random_model_accuracy
        random_model_average_accuracy = random_model_total_accuracy / small_loops
        print('Average Accuracy:', random_model_average_accuracy)
        if random_model_average_accuracy > best_average_accuracy:
            best_average_accuracy = random_model_average_accuracy
            best_features = random_list

    print('Best Average Accuracy:', best_average_accuracy)
    print('Best Features:', best_features)


if __name__ == '__main__':
    # most_important_features_measurer(100)
    regular_features_measurer(1000)