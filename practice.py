def feature_combiner(columns_to_combine=0):
    import numpy as np  # this is for doing interesting things with numbers
    import sklearn  # this is the machine learning module
    from sklearn import linear_model
    import itertools
    import functions
    import time
    from functions import seconds_formatter
    from importance_finder import feature_importer
    import pandas as pd

    df = pd.read_csv("scaled_dataframe.csv")
    most_important_features = feature_importer(22)
    dataframe = df[['G3', 'G2', 'age']]
    AllDataColumns = dataframe.columns
    AlldataframesColumnsList = AllDataColumns.tolist()
    DataPicks = dataframe[AlldataframesColumnsList]
    PickeddataframeColumns = AllDataColumns.drop("G3")
    PickeddataframeColumnsList = PickeddataframeColumns.tolist()
    newdata = dataframe[PickeddataframeColumnsList]

    TargetVariable = "G3"

    runtimes = 5  # default should be 5
    start_time = time.time()

    print('data_length:', len(dataframe.columns) - 1, 'columns')

    combination_max = (((2 ** (
                len(dataframe.columns) - 1)) * runtimes) - runtimes)  # 22 is max amount of columns to reasonably take
    print('Combination Max:', combination_max)
    time_per_1combination = 0.0013378042273516
    runtime_predictor = time_per_1combination * combination_max
    print('Time Until Completion:', runtime_predictor, 'seconds')
    seconds_formatter(runtime_predictor)

    print('\nRun feature combiner? Hit ENTER for yes')
    user_input = input()
    if user_input == '':
        pass
    else:
        quit()

    best_average_score = 0
    combinations = 0
    total_score = 0
    for loop in PickeddataframeColumnsList:
        result = itertools.combinations(PickeddataframeColumnsList, PickeddataframeColumnsList.index(loop) + 1)
        for features_being_looped in result:
            print("Features Being Looped:", list(features_being_looped))
            for i in range(runtimes):

                X = np.array(dataframe[list(features_being_looped)])
                y = np.array(dataframe[TargetVariable])

                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                            test_size=0.2)  # add in randomstate= a # to stop randomly changing your arrays

                MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
                print('Score:', MyLinearRegression.score(X_test, y_test))
                # make this add up 5 times first one should equal 0.184
                total_score = total_score + MyLinearRegression.score(X_test, y_test)

            current_average_score = total_score / runtimes
            print('Current Average Score:', current_average_score, '\n')
            total_score = 0
            if current_average_score > best_average_score:
                best_average_score = current_average_score
                best_features = features_being_looped
                print('New Best Features:', best_features)

            combinations = combinations + 1
            print('Percent Complete:', str((combinations / combination_max) * 100)[0:4], '%')

    print("Total Combinations:", combinations)
    print('Predicted Combinations:', combination_max)
    print('Best Average Score:', best_average_score)
    print('Best Features:', best_features)

    text_best_score = functions.text_file_reader('feature_combinations_data', 13, 31)

    # write to the file
    if float(text_best_score) < best_average_score:
        text_data_list = ['\n\nBest Score:', str(best_average_score), '\nBest Features:', str(best_features),
                          '\nRunthroughs:', str(runtimes), '\nTime to Run:', str(time.time() - start_time), 'seconds',
                          '\nDate Completed:', str(time.asctime())]
        string_data_list = (', '.join(text_data_list))
        functions.text_file_appender('feature_combinations_data', string_data_list)

    elapsed_time = time.time() - start_time

    if elapsed_time > 3:
        functions.email_or_text_alert('Trainer is done',
                                      'elapsed time:' + str(elapsed_time) + ' seconds', '4052198820@mms.att.net')
        print('elapsed_time:', elapsed_time, 'seconds')
        print('predicted_time:', runtime_predictor, 'seconds')


if __name__ == '__main__':
    feature_combiner()
