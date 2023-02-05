import pandas as pd
import numpy as np

def cyclical_converter(data, column_to_change, time_frame):
    if time_frame == 'months':
        # The number is the coefficient, use an online grapher to find the coefficient
        # This is the best cycle math for months (I used DESMOS graphing calc to find numbers)
        data[column_to_change] = -np.cos(0.5236 * data[column_to_change])

    if time_frame == 'day_of_month':
        # This is the best cycle for days of the month
        data[column_to_change] = np.sin(0.1044 * data[column_to_change]) # This could be more accurate

    if time_frame == 'day_of_week':
        # This is the best cycle for day of week
        data[column_to_change] = np.sin(0.446856 * data[column_to_change])

    if time_frame == 'week_of_year':
        # This is the best cycle for week of year
        data[column_to_change] = np.sin(0.06036 * data[column_to_change])

    return data

file_location = "C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/Volare_Sales_test.csv"
data = pd.read_csv(file_location)

converted_dataframe1 = cyclical_converter(data, 'Day of Month', 'day_of_month')
converted_dataframe2 = cyclical_converter(converted_dataframe1, 'Month', 'months')
converted_dataframe3 = cyclical_converter(converted_dataframe2, 'Day of Week', 'day_of_week')
converted_dataframe4 = cyclical_converter(converted_dataframe3, 'Week of Year', 'week_of_year')

# Save the result to a csv
converted_dataframe4.to_csv(file_location[:file_location.rfind('.', 0)] + '(time_converted).csv')
