import pandas as pd


def outlier_cleaner(dataframe_to_clean):
    # This function makes outliers outside the IQR equal to the IQR bounds, unless more than 5% are outside.
    # If more than 5% are outside, it simply removes the outlier values, causing a Nan to replace its value
    outlier_free_df = pd.DataFrame()
    for column in dataframe_to_clean:
        print('\ncolumn:', column)
        if dataframe_to_clean[column].dtypes == 'object':
            print('column is categorical')
            continue
        lower_limit, upper_limit = dataframe_to_clean[column].quantile([.0, 1.0])
        print('lower/upper limit:', lower_limit, '/', upper_limit)
        len_of_dropped_rows = len(dataframe_to_clean[column].loc[(dataframe_to_clean[column] > upper_limit) | (
                    dataframe_to_clean[column] < lower_limit)])
        print('len of dropped rows:', len_of_dropped_rows)
        print('len of test[column]', len(dataframe_to_clean[column]))
        row_percent_removed = len_of_dropped_rows / len(dataframe_to_clean[column])
        print('row percent moved:', row_percent_removed)
        if row_percent_removed < 0.05:
            dataframe_to_clean[column] = dataframe_to_clean[column].clip(lower_limit, upper_limit)
            print('percent removed was less than 0.05')
        else:
            dataframe_to_clean[column] = dataframe_to_clean[column].loc[
                (dataframe_to_clean[column] <= upper_limit) & (dataframe_to_clean[column] >= lower_limit)]
            print('percent removed was greater than 0.05')
        print('final dataframe to clean[column]:\n', dataframe_to_clean[column])
        outlier_free_df = pd.concat([outlier_free_df, dataframe_to_clean[column]], axis=1)
    return outlier_free_df


def null_columns_cleaner(data_to_null_clean):
    # This function gets rid of columns entirely if more than 25% of them are null
    columns_removed_list = []
    for column in data_to_null_clean.columns:
        # print('\n\n\ncolumn:', column)
        print('percent of column ' + column + ' that is null:',
              data_to_null_clean[column].isnull().sum() / (len(data_to_null_clean)))
        if (data_to_null_clean[column].isnull().sum() / (len(data_to_null_clean))) > 1.00:
            print('column with too many null values deleted:', column)
            data_to_null_clean.drop(column, inplace=True, axis=1)
            columns_removed_list.append(column)

    print(
        '\033[33m' + '\nColumns that were removed because they had too many null values after outliers were removed:\n',
        columns_removed_list)
    print('\033[39m')
    non_empty_columns_df = data_to_null_clean.fillna(data_to_null_clean.mean(numeric_only=True))
    return non_empty_columns_df


# ToDo put in a graph to add to the grapher button that shows how much data was removed from cleaning

# def null_row_cleaner(data_to_clean):
#     # This function drops rows if too many of the columns are null
#     fully_cleaned_df = data_to_clean.dropna(how='all')
#     fully_cleaned_df.reset_index(drop=True, inplace=True)
#     return fully_cleaned_df


def unnamed_column_cleaner(data_with_unnamed_columns):
    # This function clears any columns that are unnamed
    for column in data_with_unnamed_columns.columns:
        if column.startswith('Unnamed'):
            data_with_unnamed_columns.drop([column], axis=1, inplace=True)
    fully_named_df = data_with_unnamed_columns
    return fully_named_df


def full_cleaner(dataframe_to_clean):
    fully_named_df = unnamed_column_cleaner(dataframe_to_clean)
    outlier_free_df = outlier_cleaner(fully_named_df)
    non_empty_columns_df = null_columns_cleaner(outlier_free_df)
    fully_cleaned_df = non_empty_columns_df
    # null_free_df = null_row_cleaner(null_column_free_df) # not sure if I want this function
    return fully_cleaned_df


# ToDo put in a gui tool that lets you decided what iqr ranges and other cleaning parameters you want


if __name__ == '__main__':
    dataframe = pd.read_csv('multiple encoded csv')
    pd.options.display.width = 500
    pd.set_option('display.max_rows', 888888)
    pd.set_option('display.max_columns', 888888)

    fully_cleaned_df = full_cleaner(dataframe)
    print('fully cleaned df:\n', fully_cleaned_df)
