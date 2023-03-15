import numpy as np
import pandas as pd
from scipy import stats


def full_transformer(dataframe, target_variable):
    if target_variable == None:
        X = dataframe
    else:
        X = dataframe.drop([target_variable], axis=1)


    log_transformed_df = X.copy(deep=True)
    sqr_root_transformed_df = X.copy(deep=True)
    box_cox_transformed_df = X.copy(deep=True)

    raw_df_skew_score = abs(X.skew(numeric_only=True).sum())

    for column in log_transformed_df.columns:
        if log_transformed_df[column].dtypes != 'object':
            log_skew_score, log_transformed_column = log_transformer(log_transformed_df, column)
            log_transformed_df[column] = log_transformed_column
    log_df_skew_score = log_transformed_df.skew(numeric_only=True).sum()

    for column in sqr_root_transformed_df.columns:
        if sqr_root_transformed_df[column].dtypes != 'object':
            sqr_root_skew_score, sqr_root_transformed_column = sqr_root_transformer(sqr_root_transformed_df, column)
            sqr_root_transformed_df[column] = sqr_root_transformed_column
    sqr_root_df_skew_score = sqr_root_transformed_df.skew(numeric_only=True).sum()

    try:
        for column in box_cox_transformed_df.columns:
            if box_cox_transformed_df[column].dtypes != 'object':
                if float(min(box_cox_transformed_df[column])) > 0:
                    box_cox_skew_score, box_cox_transformed_column, box_cox_lambda = box_cox_transformer(
                        box_cox_transformed_df, column)
                    box_cox_transformed_df[column] = box_cox_transformed_column
        box_cox_df_skew_score = box_cox_transformed_df.skew(numeric_only=True).sum()
    except:
        box_cox_df_skew_score = 9999

    print('raw skew score:', raw_df_skew_score)
    print('log skew score:', log_df_skew_score)
    print('sqr root skew score:', sqr_root_df_skew_score)
    print('box cox skew score:', box_cox_df_skew_score)

    if raw_df_skew_score <= log_df_skew_score and raw_df_skew_score <= sqr_root_df_skew_score and raw_df_skew_score <= box_cox_df_skew_score:
        return dataframe

    if log_df_skew_score <= sqr_root_df_skew_score and log_df_skew_score <= box_cox_df_skew_score and log_df_skew_score <= raw_df_skew_score:
        return log_transformed_df

    if sqr_root_df_skew_score <= log_df_skew_score and sqr_root_df_skew_score <= box_cox_df_skew_score and sqr_root_df_skew_score <= raw_df_skew_score:
        return sqr_root_transformed_df

    if box_cox_df_skew_score <= log_df_skew_score and box_cox_df_skew_score <= sqr_root_df_skew_score and box_cox_df_skew_score <= raw_df_skew_score:
        return box_cox_transformed_df


def target_transformer(dataframe, target_variable):
    # Make copy columns so the original dataframe doesn't get changed
    target_log_transformed_df = dataframe.copy(deep=True)
    target_sqr_root_transformed_df = dataframe.copy(deep=True)
    target_box_cox_transformed_df = dataframe.copy(deep=True)

    # Get the skew scores
    target_raw_skew_score = abs(dataframe[target_variable].skew())
    target_log_skew_score, target_log_transformed_column = log_transformer(target_log_transformed_df, target_variable)
    target_sqr_root_skew_score, target_sqr_root_transformed_column = sqr_root_transformer(target_sqr_root_transformed_df, target_variable)
    if float(min(target_box_cox_transformed_df[target_variable])) > 0:
        target_box_cox_skew_score, target_box_cox_transformed_column, target_box_cox_lambda = box_cox_transformer(
            target_box_cox_transformed_df, target_variable)
        target_box_cox_skew_score = 999999
    else:
        target_box_cox_skew_score = 999999999

    # Select the least skew score
    if target_raw_skew_score <= target_log_skew_score and target_raw_skew_score <= target_sqr_root_skew_score and target_raw_skew_score <= target_box_cox_skew_score:
        target_winner = 'raw'
        return target_winner, dataframe[target_variable], None
    if target_log_skew_score <= target_sqr_root_skew_score and target_log_skew_score <= target_box_cox_skew_score and target_log_skew_score <= target_raw_skew_score:
        target_winner = 'log transformer'
        dataframe[target_variable] = target_log_transformed_column
        return target_winner, dataframe[target_variable], None
    if target_sqr_root_skew_score <= target_log_skew_score and target_sqr_root_skew_score <= target_box_cox_skew_score and target_sqr_root_skew_score <= target_raw_skew_score:
        target_winner = 'sqr root transformer'
        dataframe[target_variable] = target_sqr_root_transformed_column
        return target_winner, dataframe[target_variable], None
    if target_box_cox_skew_score <= target_log_skew_score and target_box_cox_skew_score <= target_sqr_root_skew_score and target_box_cox_skew_score <= target_raw_skew_score:
        target_winner = 'box cox transformer'
        dataframe[target_variable] = target_box_cox_transformed_column
        return target_winner, dataframe[target_variable], target_box_cox_lambda



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

    box_cox_transformed_column, box_cox_lambda = stats.boxcox(dataframe[column], lmbda=None)

    skew_score = abs(pd.Series(box_cox_transformed_column).skew())
    return skew_score, box_cox_transformed_column, box_cox_lambda


if __name__ == '__main__':
    data = pd.read_csv('C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/real_estate_data/train.csv')
    # data = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/student-mat.csv")
    target_variable = 'SalePrice'
    print('data head:\n', data.head())
    target_winner, target_transformed_column, target_box_cox_lambda = target_transformer(data, target_variable)
    print('target transformed column:\n', target_transformed_column)
    # print('skew winner:', skew_winner)
    # print('skew free df:\n', skew_free_df)
    # print('Log Back-Transformed SalePrice:\n', np.exp(skew_free_df['G3']))
