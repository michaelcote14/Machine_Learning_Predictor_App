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
import math
import pickle


# ToDo divide most important features by 5 and give them to the microprocessors

amount_of_processors = 5
pd.options.display.width = 500
original_df = pd.read_csv("scaled_dataframe.csv")
print('Original Df:\n', original_df.head())
print('Length of Original Dataframe Columns:', len(original_df.columns))
with open('most_important_features.pickle', 'rb') as f:
    most_important_features = pickle.load(f)
print('Most Important Features:', most_important_features)
most_important_df = original_df[most_important_features]
print("Most Important DF:\n", most_important_df.head())

TargetVariable = "G3"
iterator = math.ceil(len(most_important_features)/amount_of_processors)
fraction = 0
group_number = 0
group_dictionary = {}
for columns in range(math.ceil(len(most_important_features)/amount_of_processors)):
    group_dictionary[group_number] = most_important_features[fraction:iterator]
    group_number = group_number + 1
    print('columns:', most_important_features[fraction:iterator])
    fraction += math.ceil(len(most_important_features)/amount_of_processors)
    iterator += math.ceil(len(most_important_features)/amount_of_processors)

print('group dictionary:\n', group_dictionary)
group1 = group_dictionary[0]
group2 = group_dictionary[1]
group3 = group_dictionary[2]
group4 = group_dictionary[3]
group5 = group_dictionary[4]
# ToDo save these to individual groups somehow
# possible solutions:  use a dictionary
print('group1:', group1)

def feature_combiner(group):
    runtimes = 5  # default should be 5
    print('Data Length:', len(original_df.columns) - 1, 'columns')

    combination_max = (((2 ** (
                len(original_df.columns) - 1)) * runtimes) - runtimes)  # 22 is max amount of columns to reasonably take
    time_per_1combination = 0.0013378042273516
    runtime_predictor = time_per_1combination * combination_max
    print('Predicted Time to Run:', time_formatter(runtime_predictor))


    best = 0
    combinations = 0
    total_score = 0
    for loop in group:
        result = itertools.combinations(group, group.index(loop) + 1)
        for item in result:
            print("item:", list(item))
            for i in range(runtimes):
                combinations = combinations + 1
                print('combinations:', combinations)
                print('Percent Complete:', str((combinations / combination_max) * 100)[0:4], '%')

                newdata = list(item)

                X = np.array(original_df[newdata])
                y = np.array(original_df[TargetVariable])

                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                            test_size=0.2)  # add in randomstate= a # to stop randomly changing your arrays

                MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
                print('Score:', MyLinearRegression.score(X_test, y_test))
                # make this add up 5 times first one should equal 0.184
                total_score = total_score + MyLinearRegression.score(X_test, y_test)
                # ToDo fix trainer by making average score the only score measured instead of best score

            average_score = total_score / runtimes
            print('Average Score:', average_score, '\n')
            total_score = 0
            if average_score > best:
                best = average_score
                best_features = newdata
                print('newdata:', newdata)
                print('best_features:', best_features)

    print("Total Combinations:", combinations)
    print('Predicted Combinations:', combination_max)
    print('Best Score:', best)
    print('Best Features:', best_features)

    return best


if __name__ == '__main__':
    print(feature_combiner(group1))
    print(feature_combiner(group2))
    print(feature_combiner(group3))
    print(feature_combiner(group4))
    print(feature_combiner(group5))






