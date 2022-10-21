def feature_combiner():
    import numpy as np  # this is for doing interesting things with numbers
    import sklearn  # this is the machine learning module
    from sklearn import linear_model
    import itertools
    import functions
    import time
    import math
    import multiple_hot_encoder
    from functions import seconds_formatter
    from correlations import correlator

    encoded_df = multiple_hot_encoder.multiple_encoder()
    top_correlators = correlator()
    dataframe = encoded_df[top_correlators]
    print("All dataframe Columns:", dataframe.columns)
    AllDataColumns = dataframe.columns
    AlldataframesColumnsList = AllDataColumns.tolist()
    DataPicks = dataframe[AlldataframesColumnsList]
    PickeddataframeColumns = AllDataColumns.drop("G3")
    PickeddataframeColumnsList = PickeddataframeColumns.tolist()
    newdata = dataframe[PickeddataframeColumnsList]

    TargetVariable = "G3"

    best = 0
    combinations = 0
    runtimes = 1 # default should be 5
    start_time = time.time()

    print('factorial:', (math.factorial(len(dataframe.columns)-3)+7)*runtimes)
    print('data_length:', len(dataframe.columns)-1, 'columns')

    combination_max = (((2**(len(dataframe.columns)-1))*runtimes)-runtimes) # 22 is max amount of columns to reasonably take
    print('Combination Max:', combination_max)
    time_per_1combination = 0.0013378042273516
    runtime_predictor = time_per_1combination * combination_max
    print('Time Until Completion:', runtime_predictor, 'seconds')
    seconds_formatter(runtime_predictor)

    print('\nRun feature iterator? Hit ENTER for yes')
    user_input = input()
    if user_input == '':
        pass
    else:
        quit()


    total_score = 0
    for loop in PickeddataframeColumnsList:
        result = itertools.combinations(PickeddataframeColumnsList, PickeddataframeColumnsList.index(loop)+1)
        for item in result:
            print("item:", list(item))
            for i in range(runtimes):
                combinations = combinations + 1
                print('combinations:', combinations)
                print('Percent Complete:', str((combinations/combination_max)*100)[0:4], '%')

                newdata = list(item)


                X = np.array(dataframe[newdata])
                y = np.array(dataframe[TargetVariable])

                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)  # add in randomstate= a # to stop randomly changing your arrays

                MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)
                print('Score:', MyLinearRegression.score(X_test, y_test))
                # make this add up 5 times first one should equal 0.184
                total_score = total_score + MyLinearRegression.score(X_test, y_test)
                my_linear_regression_score = MyLinearRegression.score(X_test, y_test)

                if my_linear_regression_score > best:
                    best = my_linear_regression_score
                    print('newdata', newdata)
                    best_features = newdata
                    print('best_features', best_features)

            average_score = total_score/runtimes
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

    print("feature_combinations_data's Best Score:")
    text_best_score = functions.text_file_reader('feature_combinations_data', 13, 31)


    # write to the file
    if float(text_best_score) < best:
        text_data_list = ['\n\nBest Score:', str(best), '\nBest Features:',  str(best_features),
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



if __name__ == '__main__':
    feature_combiner()
