import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 88)
original_dataframe = pd.read_csv("../Saved_CSVs/nfl_data(numeric_only).csv")

# dataframe.hist()
# plt.show()

# How to check the skew of a dataset (you can also do independent columns)
# print(data.skew())

def full_transformer(dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtypes != 'object':
            print('\ncolumn:', column)
            raw_skew_score = dataframe[column].skew()
            print('raw skew score:', raw_skew_score)
            log_skew_score, log_transformed_column = log_transformer(dataframe, column)
            print('log skew score:', log_skew_score)
            sqr_root_skew_score, sqr_root_transformed_column = sqr_root_transformer(dataframe, column)
            print('sqr root skew score:', sqr_root_skew_score)
            if float(min(dataframe[column])) > 0:
                box_cox_skew_score, box_cox_transformed_column = box_cox_transformer(dataframe, column)
            else:
                box_cox_skew_score = 500
            print('box cox skew score:', box_cox_skew_score)
            if raw_skew_score < log_skew_score and raw_skew_score < sqr_root_skew_score and raw_skew_score < box_cox_skew_score:
                winner = 'raw'
            if log_skew_score < sqr_root_skew_score and log_skew_score < box_cox_skew_score and log_skew_score < raw_skew_score:
                winner = 'log transformer'
                dataframe[column] = log_transformed_column
            if sqr_root_skew_score < log_skew_score and sqr_root_skew_score < box_cox_skew_score and sqr_root_skew_score < raw_skew_score:
                winner = 'sqr root transformer'
                dataframe[column] = sqr_root_transformed_column
            if box_cox_skew_score < log_skew_score and box_cox_skew_score < sqr_root_skew_score and box_cox_skew_score < raw_skew_score:
                winner = 'box cox transformer'
                dataframe[column] = box_cox_transformed_column

            print('winner:', winner)
            print('dataframe[', column, ']\n', dataframe[column])
    return dataframe


def log_transformer(dataframe, column):
    # How to log transform each column and check the skew
    minimum_value = float(min(dataframe[column]))
    if minimum_value < 0:
        log_transformed_column = np.log(abs(minimum_value) + (dataframe[column] + 1))
    else:
        log_transformed_column = np.log(dataframe[column] + 1)
    skew_score = abs(log_transformed_column.skew())

    return skew_score, log_transformed_column

def sqr_root_transformer(dataframe, column):
    # How to square root transform and check the skew
    minimum_value = float(min(dataframe[column]))
    if minimum_value < 0:
        sqr_root_transformed_column = np.sqrt(abs(minimum_value) + (dataframe[column] + 1))
    else:
        sqr_root_transformed_column = np.sqrt(dataframe[column] + 1)

    skew_score = abs(sqr_root_transformed_column.skew())
    return skew_score, sqr_root_transformed_column

def box_cox_transformer(dataframe, column):
    # How to transform using the boxcox method
    minimum_value = float(min(dataframe[column]))
    box_cox_transformed_column = stats.boxcox(dataframe[column])[0]

    skew_score = abs(pd.Series(box_cox_transformed_column).skew())
    return skew_score, box_cox_transformed_column

test_dataframe = original_dataframe.copy()
unskewed_df = full_transformer(test_dataframe)
# print('unskewed_df:\n', unskewed_df)
# print('original dataframe skew:', abs(original_dataframe.skew(numeric_only=True)).sum())
# print('unskewed_df skew:', abs(unskewed_df.skew(numeric_only=True)).sum())

def tester(dataframe):
    # Scoring Tester
    target_variable = 'Wind_Speed'

    X = np.array(dataframe.drop([target_variable], axis=1))
    y = np.array(dataframe[target_variable])

    regression_line = linear_model.LinearRegression()

    total_score = 0
    for lap in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        regression_line.fit(X_train, y_train)
        score = regression_line.score(X_test, y_test)
        total_score = total_score + score

    average_score = total_score/1000

    return average_score

print('\noriginal dataframe score:', tester(original_dataframe))
print('unskewed_df score:', tester(unskewed_df))


























