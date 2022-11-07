import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
import time
import functions
import pandas as pd
import concurrent.futures


dataframe = pd.read_csv('scaled_dataframe.csv')
print('---------------------dataframe--------------------\n', dataframe)

data = dataframe[['G3', 'G2', 'G1', 'reason_home', 'nursery_yes', 'health', 'failures', 'school_MS', 'Pstatus_T', 'reason_reputation', 'Mjob_services', 'famrel', 'reason_other']]
target_variable = 'G3'

X = np.array(data.drop([target_variable], axis=1))
y = np.array(data[target_variable])

# ToDo find a way to predict time
runtimes, new_runtimes = 10, 10

def main_trainer(runtimes):
    start_time = time.time()
    for j in range(runtimes):
        current_model_total_accuracy, old_pickle_model_total_accuracy, updates_to_pickle_model = 0, 0, 0
        for i in range(new_runtimes):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            current_model_regression_line = linear_model.LinearRegression()
            current_model_regression_line.fit(X_train, y_train)
            current_model_accuracy = current_model_regression_line.score(X_test, y_test)
            current_model_total_accuracy = current_model_total_accuracy + current_model_accuracy
            with open(('Data/studentmodel.pickle', 'rb')) as f:
                old_pickled_regression_line = pickle.load(f)
            old_pickle_model_accuracy = old_pickled_regression_line.score(X_test, y_test)
            old_pickle_model_total_accuracy += old_pickle_model_accuracy

        current_model_average_accuracy = current_model_total_accuracy/new_runtimes
        print('\nCurrent Model Average Accuracy:', current_model_average_accuracy)
        old_pickle_model_average_accuracy = old_pickle_model_total_accuracy / new_runtimes
        print('Old Pickle Model Average Accuracy', old_pickle_model_average_accuracy)

        if current_model_average_accuracy > old_pickle_model_average_accuracy:
            updates_to_pickle_model += updates_to_pickle_model
            with open('Data/studentmodel.pickle', 'wb') as f:
                pickle.dump(current_model_regression_line, f)

    trainer_data_saved_score = functions.text_file_reader('trainer_data.txt', 13, 31)
    print('Saved File Score:', trainer_data_saved_score)

    if old_pickle_model_average_accuracy > float(trainer_data_saved_score):
        list_to_store = ['\n\n Pickle Average Accuracy:', str(old_pickle_model_average_accuracy), '\n Date:',
        str(time.asctime()), 'Features Used:', str(data.columns[1:1000].tolist())]

        stored_list_to_string = (', '.join(list_to_store))
        functions.text_file_appender('trainer_data.txt', stored_list_to_string)


    print('\033[32m', 'Updates to Pickle Model:', updates_to_pickle_model), print('\033[0m')
    return updates_to_pickle_model






if __name__ == '__main__':
    processor_runs = 100000
    print('Predicted Run Time:', functions.time_formatter(0.0029252289056777953 * processor_runs))
    user_input = input('Do you want to run Feature Combiner? Hit ENTER for Yes: ')
    if user_input == '':
        pass
    else:
        quit()
    multiprocessor_start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # this is how you get a return from multiprocessing
        f1 = executor.submit(main_trainer, int(processor_runs/5))
        f2 = executor.submit(main_trainer, int(processor_runs/5))
        f3 = executor.submit(main_trainer, int(processor_runs/5))
        f4 = executor.submit(main_trainer, int(processor_runs/5))
        f5 = executor.submit(main_trainer, int(processor_runs/5))


        print('Current Model Average Accuracy:', f1.result())
        multiprocessor_elapsed_time = time.time() - multiprocessor_start_time




    total_pickle_model_updates = f1.result() + f2.result() + f3.result() + f4.result() + f5.result()
    print('Total Pickle Model Updates:', total_pickle_model_updates)
    print('Predicted Run Time:', functions.time_formatter(0.0023252289056777953 * processor_runs))
    print('\033[34m', 'Multiprocessor Elapsed Time:', functions.time_formatter(multiprocessor_elapsed_time)), print('\033[0m')

    if multiprocessor_elapsed_time > 30:
        functions.email_or_text_alert('Trainer:', 'Total Pickle Model Updates:' + str(total_pickle_model_updates),
                                      '4052198820@mms.att.net')







