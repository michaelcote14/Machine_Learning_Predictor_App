import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import linear_model
from matplotlib import pyplot as plt



def predictor(dataframe, target_variable):

    # Splits the data
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for train_index, test_index in split.split(dataframe, dataframe[[target_variable, 'pass_rating']]):
        stratified_training_set = dataframe.loc[train_index]
        stratified_testing_set = dataframe.loc[test_index]

    X_train = stratified_training_set.drop([target_variable], axis=1)
    y_train = stratified_training_set[target_variable]

    X_test = stratified_testing_set.drop([target_variable], axis=1)
    y_test = stratified_testing_set[target_variable]

    regression_line = linear_model.LinearRegression()
    regression_line.fit(X_train, y_train)
    score = regression_line.score(X_test, y_test)
    print('Score:', score)

    # regression_line.fit(X_train2, y_train2)
    # score2 = regression_line.score(X_test2, y_test2)
    # print('Score2:', score2)



    # # Make dataframe from selected features
    # shortened_dataframe = dataframe[target_variable]
    #
    # X = np.array(shortened_dataframe.drop([self.target_variable], axis=1), dtype='object')
    # y = np.array(shortened_dataframe[self.target_variable], dtype='object')
    #
    # # ToDo put in a quick predictor? Using the line below
    # # finalized_predictor_array = Predictor.predictor_array_cleaner(self, shortened_dataframe, self.target_variable)
    # pickle_in = open('saved_training_pickle_models/' + self.selected_training_model + '.pickle', 'rb')
    # regression_line = pickle.load(pickle_in)
    #
    # runtimes = 100  # ToDo what should default runtimes be?
    #
    # total_predictions = total_score = total_mean_absolute_error = 0
    # for i in range(runtimes):
    #     if self.is_data_split == 0:
    #         X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    #
    #     if self.is_data_split == 0:
    #         predictions = regression_line.predict(X_test)
    #         score = regression_line.score(X_test, y_test)
    #         mean_absolute_error = metrics.mean_absolute_error(y_test, predictions)
    #
    #     else:
    #         predictions = regression_line.predict(X)
    #         score = regression_line.score(X, y)
    #         mean_absolute_error = metrics.mean_absolute_error(y, predictions)
    #
    #     total_predictions = np.add(predictions, total_predictions)
    #     total_score += score
    #     total_mean_absolute_error += mean_absolute_error
    #
    #     # self.data_known_prediction = regression_line.predict(finalized_predictor_array)[0]
    #     # ToDo make sure raw doesn't win everytime in the future
    #
    # self.average_predictions = total_predictions / runtimes
    #
    # self.average_score = total_score / runtimes
    # self.average_mae = total_mean_absolute_error / runtimes
    #
    # if self.is_data_split == 0:
    #     self.all_actual_values = y_test
    # else:
    #     self.all_actual_values = y
    #
    # return self.average_predictions, self.all_actual_values

# data = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Data/real_estate_data/train.csv")
data = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Data/nfl_data(shortened_numerical).csv")
print(data.head())

target_variable = 'Wind_Speed'

predictor(data, target_variable)