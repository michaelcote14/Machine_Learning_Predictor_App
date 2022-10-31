import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
import time
import functions
import pandas as pd

def main_trainer():
    dataframe = pd.read_csv('scaled_dataframe.csv')
    print('---------------------dataframe--------------------', dataframe)

    data = dataframe[['G3', 'G2', 'G1', 'reason_home', 'nursery_yes', 'health', 'failures', 'school_MS', 'Pstatus_T', 'reason_reputation', 'Mjob_services', 'famrel', 'reason_other']]
    target_variable = 'G3'

    X = np.array(data.drop([target_variable], axis=1))
    y = np.array(data[target_variable])

    runtimes, new_runtimes = 10, 10
    predicted_time = 0.0010101366043091 * runtimes * new_runtimes
    print('Predicted Time:', predicted_time, 'seconds')
    functions.trainer_runtime_predictor(predicted_time)

    print('Run Trainer? Hit ENTER for yes')
    user_input = input()
    if user_input == '':
        pass
    else:
        quit()

    start_time = time.time()
    combination_count, updates_to_pickle_model = 0, 0
    for j in range(runtimes):
        old_pickle_model_total_accuracy, current_model_total_accuracy = 0, 0
        for i in range(new_runtimes):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            current_model_regression_line = linear_model.LinearRegression()
            current_model_regression_line.fit(X_train, y_train)
            current_model_accuracy = current_model_regression_line.score(X_test, y_test)
            current_model_total_accuracy += current_model_accuracy
            current_model_average_accuracy = current_model_total_accuracy/new_runtimes

            old_pickled_regression_line = pickle.load(open('Data/studentmodel.pickle', 'rb'))
            old_pickle_model_accuracy = old_pickled_regression_line.score(X_test, y_test)
            old_pickle_model_total_accuracy += old_pickle_model_accuracy
            old_pickle_model_average_accuracy = old_pickle_model_total_accuracy/new_runtimes
            combination_count = combination_count + 1
            print('Percent Complete:', str((combination_count / (runtimes * new_runtimes) * 100))[0:5], '%')

            if current_model_average_accuracy > old_pickle_model_average_accuracy:
                updates_to_pickle_model += updates_to_pickle_model
                with open('Data/studentmodel.pickle', 'wb') as f:
                    pickle.dump(current_model_regression_line, f)


    elapsed_time = time.time() - start_time
    print('\nPredicted Time:', predicted_time, 'seconds OR', end=' ')
    functions.seconds_formatter(predicted_time)
    print('Actual Time:', elapsed_time, 'seconds OR', end=' ')
    functions.seconds_formatter(elapsed_time), print('\n')
    trainer_data_saved_score = functions.text_file_reader('trainer_data.txt', 13, 31)
    print('Saved File Score:', trainer_data_saved_score)
    if str(old_pickle_model_average_accuracy) > trainer_data_saved_score:
        list_to_store = ['\n\n Pickle Average Accuracy:', str(old_pickle_model_average_accuracy), '\n Date:',
        str(time.asctime()), 'Time to Run:', str(functions.seconds_formatter(elapsed_time)), 'Features Used:',
        str(data.columns[1:1000].tolist())]

        stored_list_to_string = (', '.join(list_to_store))
        functions.text_file_appender('trainer_data.txt', stored_list_to_string)

    if elapsed_time > 30:
        functions.email_or_text_alert('Trainer:', 'Pickle Average Accuracy:' + str(old_pickle_model_average_accuracy),
                                      '4052198820@mms.att.net')
    print('\033[32m' + 'Updates to Pickle Model:', updates_to_pickle_model), print('\033[39m')


if __name__ == '__main__':
    main_trainer()








