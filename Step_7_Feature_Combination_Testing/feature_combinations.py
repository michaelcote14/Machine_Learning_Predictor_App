import numpy as np 
import sklearn  
from sklearn import linear_model
import itertools
from Extras import functions
import time
from Extras.functions import time_formatter
import pandas as pd
import pickle
from Step_4_Data_Cleaning.data_cleaner import target_variable

df = pd.read_csv("../Step_5_Scaling/scaled_dataframe.csv")
with open('../Data/most_important_features.pickle', 'rb') as f:
    most_important_features = pickle.load(f)[0:15]
dataframe = df[most_important_features]
print("All Dataframe Columns:", dataframe.columns.tolist())
all_data_columns = dataframe.columns
picked_dataframe_columns = all_data_columns.drop("G3")
picked_dataframe_columns_list = picked_dataframe_columns.tolist()


runtimes = 5  # default should be 5
start_time = time.time()

print('Data Length:', len(dataframe.columns) - 1, 'columns')

combination_max = (((2 ** (
            len(dataframe.columns) - 1)) * runtimes) - runtimes)  # 22 is max amount of columns to reasonably take
time_per_1combination = 0.0013378042273516
runtime_predictor = time_per_1combination * combination_max
print('Predicted Time to Run:', time_formatter(runtime_predictor))

user_input = input('Run Feature Iterator? Hit ENTER for yes: ')
if user_input == '':
    pass
else:
    quit()

def feature_combiner():


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


    print("Total Combinations:", combinations)
    print('Predicted Combinations:', combination_max)
    print('Best Average Score:', best_average_score)
    print('Best Features:', best_features)

    text_best_score = functions.text_file_reader('feature_combinations_data', 13, 31)


    # write to the file
    if float(text_best_score) < best_average_score:
        text_data_list = ['\n\nBest Average Score:', str(average_score), '\nBest Features:',  str(best_features),
        '\nRunthroughs:', str(runtimes), '\nTime to Run:', str(time.time()-start_time), 'seconds',
        '\nDate Completed:', str(time.asctime())]
        string_data_list = (', '.join(text_data_list))
        functions.text_file_appender('feature_combinations_data', string_data_list)

    elapsed_time = time.time() - start_time

    if elapsed_time > 3:
        functions.email_or_text_alert('Trainer is done',
        'elapsed time:' + str(elapsed_time) + ' seconds', '4052198820@mms.att.net')
        print('elapsed_time:', elapsed_time, 'seconds')
        print('predicted_time:', runtime_predictor, 'seconds')
    return best_features, best_average_score


if __name__ == '__main__':
    start_time = time.time()
    feature_combiner()


    print('Elapsed Time:', time.time() - start_time)
