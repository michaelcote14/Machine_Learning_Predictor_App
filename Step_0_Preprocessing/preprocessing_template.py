import numpy as np
import pandas as pd

train_file_name = "real_estate_data/train.csv"
test_file_name = "real_estate_data/test.csv"
train_df = pd.read_csv(
    "C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_Starting_Dataframes/" + train_file_name)
test_df = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_Starting_Dataframes/" + test_file_name)
print(test_df)

# Remove test_df Id column, so it can be added back in later
id_dataframe = test_df['Id']
test_df = test_df.drop('Id', axis=1)
print(test_df)

# Check all column dtypes
for column in train_df.columns:
    print(column + " : ", train_df[column].dtypes)

# Sort train_df alphabetically
train_df = train_df.sort_index(axis=1)
test_df = test_df.sort_index(axis=1)

# Remove unnamed columns
for column in train_df.columns:
    if column.startswith('Unnamed'):
        train_df.drop([column], axis=1, inplace=True)
        test_df.drop([column], axis=1, inplace=True)

# Convert all strings to uppercase
for column in train_df.columns:
    if train_df[column].dtypes == 'object':
        train_df[column] = train_df[column].str.upper()
        test_df[column] = test_df[column].str.upper()

        # Clears the white spaces in all the rows
        train_df[column].str.strip()
        test_df[column].str.strip()

# Clean spaces in column names and add in underscore spaces
train_df.columns = train_df.columns.str.lstrip()
test_df.columns = test_df.columns.str.lstrip()
train_df.columns = train_df.columns.str.rstrip()
test_df.columns = test_df.columns.str.rstrip()

# Replace spaces in middle of text with underscore
train_df.columns = train_df.columns.str.replace(' ', '_')
test_df.columns = test_df.columns.str.replace(' ', '_')

# Replace boolean values with strings
for column in train_df.columns:
    if train_df.dtypes[column] == 'bool':
        train_df[column] = train_df[column].astype(str)
        test_df[column] = test_df[column].astype(str)

# Remove duplicates
train_df.drop_duplicates()
test_df.drop_duplicates()

# Replace fake nans with real nans
# train_df.replace(to_replace='Missing', value=np.nan, inplace=True)
# test_df.replace(to_replace='Missing', value=np.nan, inplace=True)

# Splice values from a certain column
# train_df['Year'] = train_df['Year'].astype('str')
# train_df['Year'] = train_df['Year'].str[0:2]
# train_df['Year']

# test_df['Year'] = test_df['Year'].astype('str')
# test_df['Year'] = test_df['Year'].str[0:2]
# test_df['Year']

# Split off certain characters of a column
# train_df['Date'].apply(lambda x: str(x).split('/')[0])
# test_df['Date'].apply(lambda x: str(x).split('/')[0])

# Remove all numbers in a cell using regex
# train_df['Date'] = train_df['Date'].str.replace('\d+', '')
# test_df['Date'] = test_df['Date'].str.replace('\d+', '')

# Convert numerical column to categorical
train_df['MSSubClass'] = train_df['MSSubClass'].astype(str)  # can use ordered=True if numerically related
test_df['MSSubClass'] = test_df['MSSubClass'].astype(str)  # can use ordered=True if numerically related


# Can change the category names using syntax below
# categorical_temp_column = categorical_temp_column.rename_categories(['Category1', 'Category2', 'Category3'])
# test_categorical_temp_column = test_categorical_temp_column.rename_categories(['Category1', 'Category2', 'Category3'])

# Scale cyclical data to proper format
def cyclical_converter(train_df, column_to_change, time_frame):
    if time_frame == 'day_of_month':
        # This is the best cycle for days of the month
        train_df[column_to_change] = np.sin(0.1044 * train_df[column_to_change])  # This could be more accurate

    if time_frame == 'day_of_week':
        # This is the best cycle for day of week
        train_df[column_to_change] = np.sin(0.446856 * train_df[column_to_change])

    if time_frame == 'week_of_year':
        # This is the best cycle for week of year
        train_df[column_to_change] = np.sin(0.06036 * train_df[column_to_change])

    if time_frame == 'month_of_year':
        # The number is the coefficient, use an online grapher to find the coefficient
        # This is the best cycle math for months (I used DESMOS graphing calc to find numbers)
        train_df[column_to_change] = -np.cos(0.5236 * train_df[column_to_change])

    return train_df


train_converted_dataframe1 = cyclical_converter(train_df, 'MoSold', 'month_of_year')
# train_converted_dataframe2 = cyclical_converter(converted_dataframe1, 'Month', 'months')
# train_converted_dataframe3 = cyclical_converter(converted_dataframe2, 'Day of Week', 'day_of_week')
# train_converted_dataframe4 = cyclical_converter(converted_dataframe3, 'Week of Year', 'week_of_year')

test_converted_dataframe1 = cyclical_converter(test_df, 'MoSold', 'month_of_year')

# Put id column back into test train_df at the front
test_converted_dataframe1.insert(0, 'Id', id_dataframe)

pd.set_option('display.max_columns', 80)

# Save everything to pickle
train_converted_dataframe1.to_pickle(
    "C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_Starting_Dataframes/" + train_file_name + "(preprocessed).pickle")
test_converted_dataframe1.to_pickle(
    "C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_Starting_Dataframes/" + test_file_name + "(preprocessed).pickle")
