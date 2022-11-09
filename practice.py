import numpy as np
import sklearn
from sklearn import linear_model
import itertools
import functions
import time
from functions import time_formatter
from importance_finder import feature_importer_non_printing
import pandas as pd
import concurrent.futures
import pickle


# ToDo divide most important features by 5 and give them to the microprocessors
# ToDo fix time predictor
# ToDo add in input requirement to start after time predictor

amount_of_processors = 5
target_variable = "G3"
pd.options.display.width = 500
scaled_df = pd.read_csv("scaled_dataframe.csv")
print('Original Df:\n', scaled_df.head())
print('Length of Original Dataframe Columns:', len(scaled_df.columns))
with open('most_important_features.pickle', 'rb') as f:
    full_most_important_features = pickle.load(f)[0:5]
most_important_features = full_most_important_features.copy()
print('Full Most Important Features:', full_most_important_features)
most_important_features.remove(target_variable)
print('Most Important Features', most_important_features)
most_important_df = scaled_df[full_most_important_features]
print("Most Important DF:\n", most_important_df.head())


def feature_combiner(runtimes=5):
    print('Data Length:', len(most_important_features) - 1, 'columns')

    combination_max = (((2 ** (
                len(most_important_features) - 1)) * runtimes) - runtimes)  # 22 is max amount of columns to reasonably take
    time_per_1combination = 0.0013378042273516
    runtime_predictor = time_per_1combination * combination_max
    print('Predicted Time to Run:', time_formatter(runtime_predictor))


    best_average_score = 0
    combinations = 0
    total_score = 0
    for loop in most_important_features:
        result = itertools.combinations(most_important_features, most_important_features.index(loop) + 1)
        for item in result:
            print("item:", list(item))
            for i in range(runtimes):
                combinations = combinations + 1
                print('combinations:', combinations)
                print('Percent Complete:', str((combinations / combination_max) * 100)[0:4], '%')

                newdata = list(item)

                X = np.array(most_important_df[newdata])
                y = np.array(most_important_df[target_variable])

                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                            test_size=0.2)  # add in randomstate= a # to stop randomly changing your arrays

                MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
                score = MyLinearRegression.score(X_test, y_test)
                print('Score:', score)
                return score



    # print("Total Combinations:", combinations)
    # print('Predicted Combinations:', combination_max)
    # print('Best Average Score:', best_average_score)
    # print('Best Features:', best_features)
    #
    # return best_average_score, best_features

# ToDo use the microprocessors to get the average score for each column iteration (use dictionary?)


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        best_average_score1 = executor.submit(feature_combiner, 1)
        best_average_score2 = executor.submit(feature_combiner, 1)
        best_average_score3 = executor.submit(feature_combiner, 1)
        best_average_score4 = executor.submit(feature_combiner, 1)
        best_average_score5 = executor.submit(feature_combiner, 1)

    print('\033[34m', best_average_score1.result()), print('\033[0m')
    print('\033[34m', best_average_score2.result()), print('\033[0m')
    print('\033[34m', best_average_score3.result()), print('\033[0m')
    print('\033[34m', best_average_score4.result()), print('\033[0m')
    print('\033[34m', best_average_score5.result()), print('\033[0m')







