import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def full_cleaner(dataframe, target_variable=None):
    # null_free_rows_df = null_row_cleaner(duplicate_free_df)
    fully_named_df = unnamed_column_cleaner(dataframe)
    non_empty_columns_df = null_columns_cleaner(fully_named_df)
    # outlier_free_df = outlier_cleaner(non_empty_columns_df)
    from Step_1_Data_Cleaning.skew_transformer import full_skew_transformer, target_transformer
    unskewed_df = full_skew_transformer(non_empty_columns_df, target_variable)
    fully_cleaned_df = unskewed_df

    # ToDo find out how to re-add the marked out functions

    # Transform the target column and put it as an attribute to PreditorTreeviewPage
    from Step_10_Predicting.predictor import PredictorTreeviewPage
    if target_variable != None:
        target_skew_winner, transformed_target_column, target_box_cox_lambda = target_transformer(non_empty_columns_df,
                                                                                                  target_variable)
        PredictorTreeviewPage.target_skew_winner = target_skew_winner
        PredictorTreeviewPage.target_box_cox_lambda = target_box_cox_lambda

        # Recombine the target variable with the cleaned dataframe
        fully_cleaned_df[target_variable] = transformed_target_column

    # Finds out the difference between the start dataframe and the final dataframe
    original_df_columns = dataframe.columns.tolist()
    cleaned_df_columns = fully_cleaned_df.columns.tolist()
    columns_removed = list(set(original_df_columns).difference(set(cleaned_df_columns)))
    rows_removed = dataframe.shape[0] - fully_cleaned_df.shape[0]

    return fully_cleaned_df, columns_removed, rows_removed


def null_columns_cleaner(data_to_null_clean):
    # This function gets rid of columns entirely if more than 25% of them are null
    columns_removed_list = []
    for column in data_to_null_clean.columns:
        # print('percent of column ' + column + ' that is null:',
        #       data_to_null_clean[column].isnull().sum() / (len(data_to_null_clean)))
        if (data_to_null_clean[column].isnull().sum() / (len(data_to_null_clean))) > 0.25:
            data_to_null_clean.drop(column, inplace=True, axis=1)
            columns_removed_list.append(column)

    numerical_dataframe = data_to_null_clean.select_dtypes(['number'])
    categorical_dataframe = data_to_null_clean.select_dtypes(include='object')

    # Change the numeric null values to a regression-based guesser
    imputer = IterativeImputer()
    imputed_array = imputer.fit_transform(numerical_dataframe)
    imputed_df = pd.DataFrame(imputed_array, columns=numerical_dataframe.columns)

    # Change the categorical null values to the mode of the column
    for column in categorical_dataframe:
        column_mode = categorical_dataframe[column].mode()[0]
        categorical_dataframe[column].fillna(column_mode, inplace=True)

    # Recombine the categorical and numerical dataframes
    fully_imputed_df = pd.concat([categorical_dataframe, imputed_df], axis=1)

    return fully_imputed_df


def null_row_cleaner(data_to_clean):
    # This stops a pointless error from popping up
    pd.options.mode.chained_assignment = None
    # This function drops a row if every column is null
    fully_cleaned_df = data_to_clean.dropna(how='all')
    fully_cleaned_df.reset_index(drop=True, inplace=True)
    return fully_cleaned_df


def outlier_cleaner(dataframe):
    # This function makes outliers outside the IQR equal to the IQR bounds, unless more than 5% are outside.
    # If more than 5% are outside, it simply removes the outlier values, causing a Nan to replace its value
    outlier_free_df = pd.DataFrame()
    for column in dataframe:
        if dataframe[column].dtypes == 'object':
            continue

        Q1, Q3 = dataframe[column].quantile([.25, .75])
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR

        num_of_outliers = len(dataframe[column].loc[(dataframe[column] > (upper_limit)) |
                                                    (dataframe[column] < (lower_limit))])

        # Checks the percentage of outliers there are compared to the entire column. If more than 5%,
        # it will drop all the rows that contained outliers
        if (num_of_outliers / len(dataframe[column]) * 100) > 5:
            # This ensures that IQR's that are small compared to entire column will just delete the entire column
            if len(dataframe.loc[(dataframe[column] < (upper_limit)) &
                                 (dataframe[column] > (lower_limit))]) < (0.05 * len(dataframe)):
                dataframe.drop(column, axis=1, inplace=True)
            else:
                dataframe = dataframe.loc[(dataframe[column] < (upper_limit)) &
                                          (dataframe[column] > (lower_limit))]

        # If the number of outliers is less than 5% of the data, we will cap the outliers to make them fit in more
        else:
            pd.options.mode.chained_assignment = None
            dataframe[column].clip((lower_limit), (upper_limit), inplace=True)

    dataframe.reset_index(inplace=True, drop=True)

    return dataframe


def test_df_empty_row_averager(train_df, test_df):
    # This replaces each null value and replaces it with its column (mean + median) / 2
    for column in train_df.columns:
        if column in test_df.columns:
            pass
        else:
            if train_df[column].dtypes != 'object':
                column_mean = train_df[column].mean(numeric_only=True)
                column_median = train_df[column].median(numeric_only=True)
                test_df[column] = (column_mean + column_median) / 2
            else:
                # This replaces each null value in categorical columns with the mode of its column
                column_mode = train_df[column].mode()[0]
                test_df[column] = column_mode

    return test_df


def unnamed_column_cleaner(data_with_unnamed_columns):
    # This function clears any columns that are unnamed
    for column in data_with_unnamed_columns.columns:
        if column.startswith('Unnamed'):
            data_with_unnamed_columns.drop([column], axis=1, inplace=True)
    fully_named_df = data_with_unnamed_columns
    return fully_named_df


# ToDo put in a gui tool that lets you decided what iqr ranges and other cleaning parameters you want


if __name__ == '__main__':
    pd.set_option('display.max_rows', 888888)
    # pd.set_option('display.max_columns', 888888)

    # Real estate data
    dataframe = pd.read_csv('C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/nfl_data.csv')

    # Nfl shortened data
    # dataframe = pd.read_csv(
    #     'C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Data/nfl_data(shortened_column_cleaner).csv')

    # Train data
    # dataframe = pd.read_csv("/Saved_CSVs/(test)_train.csv")

    # fully_cleaned_df = full_cleaner(dataframe)
    #
    # print('\ndataframe shape:', dataframe.shape)
    # print('fully cleaned df shape:', fully_cleaned_df.shape)
    # print('null values\n', fully_cleaned_df.isnull().sum())

    #
    # # # Create a test dataframe
    # dataframe = pd.DataFrame({'Name': ['Greg', 'Allen', None, 'Ed', 'Steve', 'Michael',
    #                                    'Thomas', 'James', 'Martin', 'Brittany', 'Jessica', 'Amy', 'Tanya', 'Ashley',
    #                                    'Ed', 'Bill', 'Jo', 'Jamie', 'Tory', 'Tiffany', 'Isabel', 'London', 'Abby',
    #                                    'Edward', 'Ian', 'Zane', 'Cade', 'Haley', 'Kate', 'Katie', 'Raven', 'Tony',
    #                                    'Brett', 'Brent', 'Zach', 'Dave', 'Jamal', 'Nick'],
    #                           'Age': [-80, 104, None, 27, 18, 19, 21, 30, 17, 21, 24, 25, 26, 21,
    #                                   22, 27, 28, 20, 20, 23, 23, 22, 27, 24, 25, 25, 23, 18, 19, 20, 22, 25, 28, 22,
    #                                   16, 17, 22, 21]})

    full_cleaner(dataframe)
