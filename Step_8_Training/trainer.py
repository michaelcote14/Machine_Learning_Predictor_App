import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import time
from Extras import functions
import pandas as pd
import concurrent.futures
from master import target_variable
import ast

processor_runs = 100000


def feature_grabber():
    user_input = input('Use most recent pickled important features? (y=yes n=no): ')
    if user_input.lower() == 'y':
        pickle_in = open('../Data/most_recent_important_features.pickle', 'r+b')
        most_important_features = pickle.load(pickle_in)

    else:
        with open('Step_6_Feature_Importance_Finding/importance_finder_log', 'r') as file:
            most_important_features = ast.literal_eval(file.readlines()[1][25:-1])
    return most_important_features



with open('trainer_log.txt', 'r') as file:
    trainer_data_saved_score = file.readlines()[0][30:]
print('Trainer Data Saved Score:', trainer_data_saved_score)


small_loops = 10
def main_trainer(runtimes, most_important_features, predicted_time):
    start_time = time.time()


    scaled_dataframe = pd.read_csv('../Data/scaled_dataframe.csv')
    df = scaled_dataframe[most_important_features]

    X = np.array(df.drop([target_variable], axis=1))
    y = np.array(df[target_variable])
    save_pickle_to = '../Data/trained_model.pickle'

    updates_to_pickle_model = 0
    for j in range(runtimes):
        current_model_total_accuracy, old_pickled_model_total_accuracy = 0, 0
        for i in range(small_loops):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            current_model_regression_line = linear_model.LinearRegression()
            current_model_regression_line.fit(X_train, y_train)
            current_model_accuracy = current_model_regression_line.score(X_test, y_test)
            current_model_total_accuracy = current_model_total_accuracy + current_model_accuracy

            print('\033[35m', '% Complete:', format((((time.time() - start_time)/predicted_time) * 100), '.2f')), print('\033[0m')
            try:
                pickle_in = open(save_pickle_to, 'r+b')
                old_pickled_regression_line = pickle.load(pickle_in)
                old_pickled_model_accuracy = old_pickled_regression_line.score(X_test, y_test)
            except Exception as e:
                old_pickled_model_accuracy = 0
                print('\033[34m', e, '-------------------'), print('\033[0m')
            old_pickled_model_total_accuracy += old_pickled_model_accuracy

        current_model_average_accuracy = current_model_total_accuracy/small_loops
        print('\nCurrent Model Average Accuracy:', current_model_average_accuracy)
        old_pickled_model_average_accuracy = old_pickled_model_total_accuracy / small_loops
        print('Old Pickle Model Average Accuracy', old_pickled_model_average_accuracy)

        if current_model_average_accuracy > old_pickled_model_average_accuracy:
            updates_to_pickle_model = updates_to_pickle_model + 1
            print('\033[32m', ('-' * 60), 'Pickle Updated', ('-' * 60)), print('\033[0m')
            with open(save_pickle_to, 'wb') as f:
                pickle.dump(current_model_regression_line, f)


    if old_pickled_model_average_accuracy > float(trainer_data_saved_score):
        list_to_store = ['\n\nPickle Model Average Accuracy:', str(old_pickled_model_average_accuracy), '\nDate Ran:',
        str(time.asctime()), '\nFeatures Used:', str(df.columns[1:1000].tolist())]

        stored_list_to_string = (' '.join(list_to_store))
        functions.text_file_appender('trainer_log.txt', stored_list_to_string)


    return updates_to_pickle_model





if __name__ == '__main__':

    most_important_features = feature_grabber()
    print('Most Important Features:', most_important_features)
    print('Length of Features:', len(most_important_features))

    microprocessors = 5
    predicted_time = .7199564681 ** len(most_important_features) * processor_runs * microprocessors
    # ToDo fix time predictor
    print('Predicted Run Time:', functions.time_formatter(predicted_time))
    user_input = input('Run Trainer? Hit ENTER for yes: ')
    if user_input == '':
        pass
    else:
        quit()


    runtimes = 100
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # this is how you get a return from multiprocessing
        f1 = executor.submit(main_trainer, int(processor_runs/microprocessors), most_important_features, predicted_time)
        f2 = executor.submit(main_trainer, int(processor_runs/microprocessors), most_important_features, predicted_time)
        f3 = executor.submit(main_trainer, int(processor_runs/microprocessors), most_important_features, predicted_time)
        f4 = executor.submit(main_trainer, int(processor_runs/microprocessors), most_important_features, predicted_time)
        f5 = executor.submit(main_trainer, int(processor_runs/microprocessors), most_important_features, predicted_time)


    total_pickle_model_updates = f1.result() + f2.result() + f3.result() + f4.result() + f5.result()
    elapsed_time = time.time() - start_time
    # ToDo make this doable with multiprocessors again
    print('\n\nTotal Pickle Model Updates:', total_pickle_model_updates)
    print('Elapsed Time:', functions.time_formatter(elapsed_time))
    print('Predicted Run Time:', functions.time_formatter(predicted_time))


    if elapsed_time > 30:
        functions.email_or_text_alert('Trainer:', 'Total Pickle Model Updates:' + str(total_pickle_model_updates),
        '4052198820@mms.att.net')
        quit()
    else:
        quit()











