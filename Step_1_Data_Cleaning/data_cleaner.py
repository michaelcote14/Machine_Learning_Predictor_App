import pandas as pd


def boolean_cleaner(dataframe):
    # This makes sure that all boolean values in the dataframe are converted to strings
    for column in dataframe.columns:
        if dataframe.dtypes[column] == 'bool':
            dataframe[column] = dataframe[column].astype(str)
    bool_free_df = dataframe

    return bool_free_df


def full_cleaner(dataframe):
    duplicate_free_df = dataframe.drop_duplicates()
    null_free_rows_df = null_row_cleaner(duplicate_free_df)
    bool_free_df = boolean_cleaner(null_free_rows_df)
    fully_named_df = unnamed_column_cleaner(bool_free_df)
    non_empty_columns_df = null_columns_cleaner(fully_named_df)
    outlier_free_df = outlier_cleaner(non_empty_columns_df)
    fully_cleaned_df = outlier_free_df

    # Finds out the difference between the start dataframe and the final dataframe
    original_df_columns = dataframe.columns.tolist()
    cleaned_df_columns = fully_cleaned_df.columns.tolist()
    columns_removed = list(set(original_df_columns).difference(set(cleaned_df_columns)))
    rows_removed = dataframe.shape[0] - fully_cleaned_df.shape[0]

    return fully_cleaned_df, columns_removed, rows_removed


def outlier_cleaner(dataframe):
    # This function makes outliers outside the IQR equal to the IQR bounds, unless more than 5% are outside.
    # If more than 5% are outside, it simply removes the outlier values, causing a Nan to replace its value
    outlier_free_df = pd.DataFrame()
    for column in dataframe:
        print('\ncolumn:', column)
        if dataframe[column].dtypes == 'object':
            print('column is categorical, so we do not clean for outliers')
            continue
        else:
            print('column is numerical, so we do clean for outliers')

        print('dataframe[' + column + ']:\n', dataframe[column])

        Q1, Q3 = dataframe[column].quantile([.25, .75])
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        print('Upper Limit/Lower Limit:', upper_limit, '/', lower_limit)

        num_of_outliers = len(dataframe[column].loc[(dataframe[column] > (upper_limit)) |
                                                    (dataframe[column] < (lower_limit))])
        print('Number of outliers in column:', num_of_outliers)
        print('% of column that is outliers:', (num_of_outliers / len(dataframe[column]) * 100), '%')

        # Checks the percentage of outliers there are compared to the entire column. If more than 5%,
        # it will drop all the rows that contained outliers
        if (num_of_outliers / len(dataframe[column]) * 100) > 5:
            print('too many outliers, so we are going to drop all rows that contain outliers entirely')
            # This ensures that IQR's that are small compared to entire column will just delete the entire column
            if len(dataframe.loc[(dataframe[column] < (upper_limit)) &
                                 (dataframe[column] > (lower_limit))]) < (0.05 * len(dataframe)):
                dataframe.drop(column, axis=1, inplace=True)
                print('column was dropped because its IQR was too small')
            else:
                dataframe = dataframe.loc[(dataframe[column] < (upper_limit)) &
                                          (dataframe[column] > (lower_limit))]
            print('dataframe after rows were dropped:\n', dataframe)

        # If the number of outliers is less than 5% of the data, we will cap the outliers to make them fit in more
        else:
            pd.options.mode.chained_assignment = None
            dataframe[column].clip((lower_limit), (upper_limit), inplace=True)
            print('dataframe[column]:\n', dataframe[column])

    dataframe.reset_index(inplace=True, drop=True)

    return dataframe


def null_columns_cleaner(data_to_null_clean):
    # This function gets rid of columns entirely if more than 25% of them are null
    columns_removed_list = []
    for column in data_to_null_clean.columns[0:30]:
        print('percent of column ' + column + ' that is null:',
              data_to_null_clean[column].isnull().sum() / (len(data_to_null_clean)))
        if (data_to_null_clean[column].isnull().sum() / (len(data_to_null_clean))) > 0.25:
            data_to_null_clean.drop(column, inplace=True, axis=1)
            columns_removed_list.append(column)

    # This replaces each null value and replaces it with its column (mean + median) / 2
    pd.set_option('display.min_rows', 240)
    for column in data_to_null_clean.columns:
        print('\nColumn:', column)
        print('Null Values:', data_to_null_clean[column].isnull().sum())
        if data_to_null_clean[column].dtype != 'object':
            column_mean = data_to_null_clean[column].mean(numeric_only=True)

            column_median = data_to_null_clean[column].median(numeric_only=True)

            data_to_null_clean[column].fillna((column_mean+column_median) / 2, inplace=True)

        # This replaces each null value in categorical columns with the mode of its column
        else:
            column_mode = data_to_null_clean[column].mode()[0]
            data_to_null_clean[column].fillna(column_mode, inplace=True)


    return data_to_null_clean
# ToDo you could make a ratio calculator that tells you what to put in each column depending on the
#  ratio that the dataframe already has for each value #very smart


def null_row_cleaner(data_to_clean):
    # This stops a pointless error from popping up
    pd.options.mode.chained_assignment = None
    # This function drops a row if every column is null
    fully_cleaned_df = data_to_clean.dropna(how='all')
    fully_cleaned_df.reset_index(drop=True, inplace=True)
    return fully_cleaned_df


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
    dataframe = pd.read_csv('C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Data/real_estate_data/train.csv')

    # Nfl shortened data
    # dataframe = pd.read_csv(
    #     'C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Data/nfl_data(shortened_column_cleaner).csv')

    # Train data
    dataframe = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Data/(test)_train.csv")

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

