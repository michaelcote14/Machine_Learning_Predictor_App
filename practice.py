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
pd.set_option('display.max_columns', 85)
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
group_dictionary = {}
print('length of features:', math.ceil(len(most_important_features)*(1/amount_of_processors)), ':', len(most_important_features))
for columns in range(math.ceil(len(most_important_features)/amount_of_processors)):
    print('columns:', most_important_features[fraction:iterator])
    fraction += math.ceil(len(most_important_features)/amount_of_processors)
    iterator += math.ceil(len(most_important_features)/amount_of_processors)
    group1 = most_important_features[fraction:iterator]
    group_number = 0
    group_dictionary['jump'] = most_important_features[fraction:iterator]
    group_number + 1
print('group dictionary:\n', group_dictionary.items())
# ToDo save these to individual groups somehow
# possible solutions:  use a dictionary


def feature_combiner(most_important_features):
    runtimes = 5 # default should be 5
    start_time = time.time()

    best = 0
    combinations = 0
    total_score = 0
    # for column in most_important_features:
    #     full_combination_list = list(itertools.combinations(most_important_features, most_important_features.index(column)+1))
    # print(len(full_combination_list))
        # for item in result:
        #     print("item:", list(item))
#             for i in range(runtimes):
#                 combinations = combinations + 1
#                 print('combinations:', combinations)
#                 print('Percent Complete:', str((combinations/combination_max)*100)[0:4], '%')
#
#                 newdata = list(item)
#
#
#                 X = np.array(original_df[newdata])
#                 y = np.array(original_df[TargetVariable])
#
#                 X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)  # add in randomstate= a # to stop randomly changing your arrays
#
#                 MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
#                 print('Score:', MyLinearRegression.score(X_test, y_test))
#                 # make this add up 5 times first one should equal 0.184
#                 total_score = total_score + MyLinearRegression.score(X_test, y_test)
#                 #ToDo fix trainer by making average score the only score measured instead of best score
#
#             average_score = total_score/runtimes
#             print('Average Score:', average_score, '\n')
#             total_score = 0
#             if average_score > best:
#                 best = average_score
#                 best_features = newdata
#                 print('newdata:', newdata)
#                 print('best_features:', best_features)
#
#
#     print("Total Combinations:", combinations)
#     print('Predicted Combinations:', combination_max)
#     print('Best Score:', best)
#     print('Best Features:', best_features)
#
#     text_best_score = functions.text_file_reader('feature_combinations_data', 13, 31)
#
#
#     # write to the file
#     if float(text_best_score) < best:
#         text_data_list = ['\n\nBest Score:', str(best), '\nBest Features:',  str(best_features),
#         '\nRunthroughs:', str(runtimes), '\nTime to Run:', str(time.time()-start_time), 'seconds',
#         '\nDate Completed:', str(time.asctime())]
#         string_data_list = (', '.join(text_data_list))
#         functions.text_file_appender('feature_combinations_data', string_data_list)
#
#     elapsed_time = time.time() - start_time
#
#     if elapsed_time > 3:
#         functions.email_or_text_alert('Trainer is done',
#         'elapsed time:' + str(elapsed_time) + ' seconds', '4052198820@mms.att.net')
#         print('elapsed_time:', elapsed_time, 'seconds')
#
#     return combinations
#
#
if __name__ == '__main__':
    feature_combiner(most_important_features=most_important_features)


