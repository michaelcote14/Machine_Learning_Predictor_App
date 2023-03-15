import pandas as pd
import numpy as np

train_file_name = "Volare_Scraped_Dataframe(Regression_Ready).csv"
test_file_name = "Volare_Scraped_Dataframe_Test.csv"
train_df = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_Dataframes/Volare/" + train_file_name)
test_df = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_Dataframes/Volare/" + test_file_name)

pd.set_option('display.max_columns', 80)

# Remove test_df Date column ,so it can be added back in later
date_dataframe = test_df['Date']
test_df = test_df.drop('Date', axis=1)
print(test_df)

# Check all column dtypes
print(train_df.dtypes)

# Make all numeric columns actually numeric
for column in train_df.columns:
    if train_df[column].dtypes == 'object':
        # Remove the commas from the numbers
        train_df[column] = train_df[column].str.replace(',', '')

        try:
            # Save the column as a numeric
            train_df[column] = pd.to_numeric(train_df[column])
        except:
            pass

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
        try:
            train_df[column] = train_df[column].str.upper()
            test_df[column] = test_df[column].str.upper()

            # Clears the white spaces in all the rows
            train_df[column].str.strip()
            test_df[column].str.strip()
        except:
            pass



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

# Convert numerical column to categorical
train_df['Year'] = train_df['Year'].astype(str) # can use ordered=True if numerically related
test_df['Year'] = test_df['Year'].astype(str) # can u

# Scale cyclical data to proper format
def cyclical_converter(train_df, column_to_change, time_frame):
    if time_frame == 'day_of_month':
        # This is the best cycle for days of the month
        train_df[column_to_change] = np.sin(0.1044 * train_df[column_to_change]) # This could be more accurate

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

print(train_df)
train_converted_dataframe1 = cyclical_converter(train_df, 'Day_of_Month', 'day_of_month')
train_converted_dataframe2 = cyclical_converter(train_df, 'Month', 'month_of_year')

test_converted_dataframe1 = cyclical_converter(test_df, 'Day_of_Month', 'day_of_month')
test_converted_dataframe2 = cyclical_converter(test_df, 'Month', 'month_of_year')

# Put date column back in to test_df
test_converted_dataframe2.insert(0, 'Date', date_dataframe)

# Drop the day of week column from test_df
test_converted_dataframe2 = test_converted_dataframe2.drop('Day_of_Week', axis=1)

print(train_converted_dataframe2)
#
# # Remove date column from both dataframes
# train_converted_dataframe2 = train_converted_dataframe2.drop('Date', axis=1)
# test_converted_dataframe2 = test_converted_dataframe2.drop('Date', axis=1)

# Save the cleaned dataframes to pickle
train_converted_dataframe2.to_pickle("C:/Users/micha/OneDrive - University of Oklahoma/College (OneDrive)/Programming/Python/Projects/LinearRegressionRepo/Saved_Dataframes/Volare/" + train_file_name + "(preprocessed).pickle")
test_converted_dataframe2.to_pickle("C:/Users/micha/OneDrive - University of Oklahoma/College (OneDrive)/Programming/Python/Projects/LinearRegressionRepo/Saved_Dataframes/Volare/" + test_file_name + "(preprocessed).pickle")

























# # Remove all commas from revenue column
# train_df['Revenue'] = train_df['Revenue'].str.replace(',', '')
#
# # Convert revenue column to float
# train_df['Revenue'] = train_df['Revenue'].astype(float)
#
# def previous_day_revenue_finder(days_prior):
#     days_prior_revenue_list = []
#     for index, daily_revenue in enumerate(train_df['Revenue']):
#         if index >= days_prior:
#             days_prior_revenue_list.append(train_df['Revenue'][index-days_prior])
#         else:
#             days_prior_revenue_list.append(0)
#
#     return days_prior_revenue_list
#
# train_df['One_Day_Prior_Revenue'] = previous_day_revenue_finder(1)
# train_df['Two_Days_Prior_Revenue'] = previous_day_revenue_finder(2)
# train_df['Three_Days_Prior_Revenue'] = previous_day_revenue_finder(3)
# train_df['Four_Days_Prior_Revenue'] = previous_day_revenue_finder(4)
# train_df['Five_Days_Prior_Revenue'] = previous_day_revenue_finder(5)
# train_df['Six_Days_Prior_Revenue'] = previous_day_revenue_finder(6)
# train_df['Seven_Days_Prior_Revenue'] = previous_day_revenue_finder(7)
#
# # Now sum all the above columns together
# train_df['Seven_Days_Prior_Summed_Revenue'] = train_df['One_Day_Prior_Revenue'] + \
#                                               train_df['Two_Days_Prior_Revenue'] + \
#                                               train_df['Three_Days_Prior_Revenue'] + \
#                                               train_df['Four_Days_Prior_Revenue'] + \
#                                               train_df['Five_Days_Prior_Revenue'] + \
#                                               train_df['Six_Days_Prior_Revenue'] + \
#                                               train_df['Seven_Days_Prior_Revenue']
