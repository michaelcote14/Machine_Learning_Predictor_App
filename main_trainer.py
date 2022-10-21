import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
import time
import functions
import multiple_hot_encoder


def main():

    dataframe = one_hot_encoder_multiple_categories.encoded_df
    print('---------------------dataframe--------------------', dataframe)

    data = dataframe[['G3', 'G2', 'G1', 'age', 'goout', 'romantic_yes', 'traveltime', 'paid_yes', 'internet_yes', 'studytime']]

    target_variable = 'G3'

    X = np.array(data.drop([target_variable], axis=1))
    y = np.array(data[target_variable])


    runtimes = 10000
    predicted_time = 0.0011102628707885742 * runtimes
    print('Predicted Time:', predicted_time, 'seconds')
    functions.trainer_runtime_predictor(predicted_time)


    print('Run Trainer? Hit ENTER for yes')
    user_input = input()
    if user_input == '':
        print('Running...')
        pass
    else:
        quit()

    start_time = time.time()
    combinations_linear_model = 0

    PickleBest, best, TotalAccuracy = 0, 0, 0
    for _ in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        TotalAccuracy += accuracy
        if accuracy > best:
            best = accuracy
            with open('Data/studentmodel.pickle', 'wb') as f:
                pickle.dump(linear, f)
            filename = 'Data/finalized_model.sav'
            pickle.dump(linear, open(filename, 'wb'))
        combinations_linear_model = combinations_linear_model + 1
        print('Percent Complete:', str(((combinations_linear_model/runtimes)*100)/2)[0:5], '%')

    PickledRegressionLine = pickle.load(open('Data/studentmodel.pickle', 'rb'))
    PickleModelAccuracy = PickledRegressionLine.score(X_test, y_test)
    print('Current Pickle Model Accuracy:', PickleModelAccuracy)
    combinations_pickle_model = 0

    TotalPickleModelAccuracy = 0
    for _ in range(runtimes):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,test_size=0.2)

        pickle_in = open('Data/studentmodel.pickle', 'rb')
        CurrentPickleModel = pickle.load(pickle_in)
        PickleModelAccuracy = CurrentPickleModel.score(X_test, y_test)
        TotalPickleModelAccuracy += PickleModelAccuracy
        combinations_pickle_model = combinations_pickle_model + 1
        print('Percent Complete:', str(((combinations_pickle_model/runtimes)*100)/2+50)[0:5], '%')

    print('v--Text Best Score--v')
    text_best_score = functions.text_file_reader('trainer_data.txt', 13, 31)

    # write to the file
    if float(text_best_score) < best:
        text_data_list = ['\n\nBest Score:', str(best),'\nPickle Model Average Accuracy:', str(TotalPickleModelAccuracy/runtimes),
                          '\nFeatures Used:',  str(data.columns[1:1000].tolist()),
        '\nRunthroughs:', str(runtimes), '\nTime to Run:', str(time.time()-start_time), 'seconds',
        '\nDate Completed:', str(time.asctime())]
        string_data_list = (', '.join(text_data_list))
        functions.text_file_appender('trainer_data.txt', string_data_list )

    elapsed_time = time.time() - start_time
    print('Predicted Time:', predicted_time, 'seconds')
    print('Actual Time:', elapsed_time, 'seconds')
    functions.seconds_formatter(elapsed_time), print('^--Actual Time--^')

    print('\nCurrent Model Average Accuracy:', TotalAccuracy/runtimes)
    print("Stored Pickle File's Average Accuracy:", TotalPickleModelAccuracy/runtimes)

    if elapsed_time > 30:
        functions.email_or_text_alert('Trainer', 'Accuracy:' + str(best),  '4052198820@mms.att.net')

if __name__ == '__main__':
    main()
