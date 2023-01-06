import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import time
from Extras import functions
import pandas as pd
from Step_2_Visualizing.visualization import target_variable
import random

scaled_dataframe = pd.read_csv('../Step_5_Scaling/scaled_dataframe.csv')
with open('../Step_6_Feature_Importance_Finding/most_important_features.pickle', 'rb') as f:
    most_important_features = pickle.load(f)[:]
print('Length of Features:', len(most_important_features))

# ToDo put in a time predictor and percentage tracker

def best_combined_features_scorer(runtimes):
    df = scaled_dataframe
    X = np.array(df.drop([target_variable], axis=1))
    y = np.array(df[target_variable])


    best_features = ['comb_pass_rush_play', 'Humidity', 'pass_int', 'Opponent_abbrev_BAL', 'Opponent_abbrev_TAM', 'off_pct', 'pass_cmp', 'rush_att', 'pass_rating', 'home_score', 'pass_long']
    combinator_total_accuracy = 0
    for _ in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        combinator_regression_line = linear_model.LinearRegression()
        combinator_regression_line.fit(X_train, y_train)
        combinator_accuracy = combinator_regression_line.score(X_test, y_test)
        print('Accuracy:', combinator_accuracy)
        combinator_total_accuracy = combinator_total_accuracy + combinator_accuracy
    combinator_average_accuracy = combinator_total_accuracy / runtimes
    print('Combinator Average Accuracy:', combinator_average_accuracy)

    return best_features, combinator_average_accuracy



def random_features_scorer(runtimes):
    df = scaled_dataframe[most_important_features]
    columns_list = df.columns.tolist()
    print('Columns List:', columns_list)
    print('Length of Column List:', len(columns_list))


    best_average_accuracy = 0
    best_features = []
    for _ in range(runtimes):
        # makes a random dataframe
        random_int = random.randint(0, len(columns_list))
        random_list = random.choices(columns_list, k=random_int)
        print('\nRandom List:', random_list)
        random_df = df[random_list]
        # measures the accuracy of this dataframe
        X = np.array(df.drop([target_variable], axis=1))
        y = np.array(df[target_variable])

        small_loops = 10
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

    print('\nBest Average Accuracy:', best_average_accuracy)
    print('Best Features:', best_features)
    return best_features, best_average_accuracy


if __name__ == '__main__':
    combinator_features, combinator_average_accuracy = best_combined_features_scorer(1000)
    best_random_features, best_random_average_accuracy = random_features_scorer(1000)

    print('\nCombinator Features:', combinator_features)
    print('Combinator Average Accuracy:', combinator_average_accuracy)

    print('Best Random Features:', best_random_features)
    print('Best Random Average Accuracy:', best_random_average_accuracy)