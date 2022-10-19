import random
import math
import time
from numba import jit
import pandas as pd
import numpy as np
import functions
import pickle
import pandas as pd #this is to read in data sheets
import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot #this allows you to make graphs
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
from matplotlib import style #this changes the style of your plot's grid
import time
import functions
from numba import jit

def main():
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        trainer()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


def trainer():
    start_time = time.time()

    dataframe = pd.read_csv('Data/student-mat.csv', sep=',')

    data = dataframe[['Fedu', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'G1', 'G2', 'G3']]

    target_variable = 'G3'

    X = np.array(data.drop([target_variable], axis=1))
    y = np.array(data[target_variable])  #

    runtimes = 10000
    functions.trainer_runtime_predictor(runtimes)
    print('Run Trainer? Hit ENTER for yes')
    user_input = input()
    if user_input == '':
        print('Running...')
        pass
    else:
        quit()

    PickleBest, best, TotalAccuracy = 0, 0, 0
    for _ in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        # print('Accuracy:',accuracy)
        TotalAccuracy += accuracy
        if accuracy > best:
            best = accuracy
            with open('Data/studentmodel.pickle', 'wb') as f:
                pickle.dump(linear, f)
            filename = 'Data/finalized_model.sav'
            pickle.dump(linear, open(filename, 'wb'))

    PickledRegressionLine = pickle.load(open('Data/studentmodel.pickle', 'rb'))
    PickleModelAccuracy = PickledRegressionLine.score(X_test, y_test)
    print('Current Pickle Model Accuracy:', PickleModelAccuracy)

    TotalPickleModelAccuracy = 0
    for _ in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

        pickle_in = open('Data/studentmodel.pickle', 'rb')
        CurrentPickleModel = pickle.load(pickle_in)
        PickleModelAccuracy = CurrentPickleModel.score(X_test, y_test)
        TotalPickleModelAccuracy += PickleModelAccuracy
        # print('Current Pickle Model Accuracy:', PickleModelAccuracy)

    print('\nCurrent Model Average Accuracy:', TotalAccuracy / runtimes)
    print("Stored Pickle File's Average Accuracy:", TotalPickleModelAccuracy / runtimes)

    print("feature_combinations_data's Best Score:")
    text_best_score = functions.text_file_reader('trainer_data.txt', 13, 31)

    # write to the file
    if float(text_best_score) < best:
        text_data_list = ['\n\nBest Score:', str(best), '\nFeatures Used:', str(data.columns),
                          '\nRunthroughs:', str(runtimes), '\nTime to Run:', str(time.time() - start_time), 'seconds',
                          '\nDate Ran:', str(time.asctime())]
        string_data_list = (', '.join(text_data_list))
        functions.text_file_appender('trainer_data.txt', string_data_list)

    elapsed_time = time.time() - start_time
    functions.seconds_formatter(elapsed_time)
    print('O', elapsed_time, 'seconds')

    if elapsed_time > 30:
        functions.email_or_text_alert('Trainer', 'Accuracy:' + str(best), '4052198820@mms.att.net')

if __name__ == '__main__':
    main()