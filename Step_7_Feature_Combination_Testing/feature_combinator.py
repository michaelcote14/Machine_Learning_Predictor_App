import numpy as np 
import sklearn  
from sklearn import linear_model
import itertools
from Extras import functions
import time
from Extras.functions import time_formatter
import pandas as pd
import ast
import pickle

def feature_combinator_log_writer(best_average_score, text_best_score, best_features, elapsed_time):
    # write to the file
    if best_average_score > float(text_best_score):
        text_data_list = ['\n\nBest Average Score:', str(best_average_score),'\nTarget Feature:', target_variable, '\nBest Features:',  str(best_features),
        '\nRuntimes:', str(runtimes), '\nElapsed Time:', str(time_formatter(elapsed_time)), 'seconds',
        '\nDate Completed:', str(time.asctime())]
        string_data_list = (' '.join(text_data_list))
        functions.text_file_appender('feature_combinator_log', string_data_list)


def feature_grabber():
    user_input = input('Use most recent pickled important features? (y=yes n=no): ')
    if user_input.lower() == 'y':
        pickle_in = open('../Data/most_recent_important_features.pickle', 'r+b')
        most_important_features = pickle.load(pickle_in)

    else:
        with open('Step_6_Feature_Importance_Finding/importance_finder_log', 'r') as file:
            most_important_features = ast.literal_eval(file.readlines()[1][25:-1])
    return most_important_features


most_important_features = feature_grabber()
print('Most Important Features:', most_important_features)

df = pd.read_csv("../Data/scaled_dataframe.csv")
dataframe = df[most_important_features]
all_data_columns = dataframe.columns


runtimes = 5  # default should be 5
start_time = time.time()

print('Data Length:', len(dataframe.columns) - 1, 'Columns')

combination_max = (((2 ** (
            len(dataframe.columns) - 1)) * runtimes) - runtimes)  # 22 is max amount of columns to reasonably take
time_per_1combination = 0.0013378042273516
predicted_time = time_per_1combination * combination_max
print('Predicted Time to Run:', time_formatter(predicted_time))


def feature_combiner(target_variable):
    picked_dataframe_columns = all_data_columns.drop(target_variable)
    picked_dataframe_columns_list = picked_dataframe_columns.tolist()
    best_average_score = 0
    combinations = 0
    total_score = 0
    for loop in picked_dataframe_columns_list:
        result = itertools.combinations(picked_dataframe_columns_list, picked_dataframe_columns_list.index(loop)+1)
        for item in result:
            print("item:", list(item))
            for i in range(runtimes):
                combinations = combinations + 1
                print('combinations:', combinations)
                print('Percent Complete:', str((combinations/combination_max)*100)[0:4], '%')

                newdata = list(item)


                X = np.array(dataframe[newdata])
                y = np.array(dataframe[target_variable])

                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)  # add in randomstate= a # to stop randomly changing your arrays

                MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
                print('Score:', MyLinearRegression.score(X_test, y_test))
                # make this add up 5 times first one should equal 0.184
                total_score = total_score + MyLinearRegression.score(X_test, y_test)

            average_score = total_score/runtimes
            print('Average Score:', average_score, '\n')
            total_score = 0
            if average_score > best_average_score:
                best_average_score = average_score
                best_features = newdata
                print('newdata:', newdata)
                print('best_features:', best_features)


    print('Best Average Score:', best_average_score)
    print('Best Features:', best_features)

    text_best_score = functions.text_file_reader('feature_combinator_log', 21, 39)
    elapsed_time = time.time() - start_time


    feature_combinator_log_writer(best_average_score, text_best_score, best_features, elapsed_time)

    if elapsed_time > 3:
        functions.email_or_text_alert('Trainer is done',
        'Elapsed Time: ' + str(time_formatter(elapsed_time)) + '\nBest Average Score: ' + str(format(best_average_score, '.4f')), '4052198820@mms.att.net')

    return best_features, best_average_score



if __name__ == '__main__':
    start_time = time.time()
    feature_combiner(target_variable)


    print('Elapsed Time:', time_formatter(time.time() - start_time))
    print('Predicted Time:', time_formatter(predicted_time))
