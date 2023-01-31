import pandas as pd
from tkinter import *
from tkinter import filedialog

data_location = filedialog.askopenfilename(initialdir='/', title='Select A CSV File',
                                           filetypes=(('csv files', '*.csv'), ))
data = pd.read_csv(data_location)
print(data)


# Plug the numerical column to categorize in
column_to_categorize = 'MSSubClass'
data[column_to_categorize] = data[column_to_categorize].astype(str)
print(data[column_to_categorize])
# concat 'category to the beginning of the text
data[column_to_categorize] = 'Category ' + data[column_to_categorize].astype(str)
print(data[column_to_categorize])

# Save the new CSV
original_csv_name = data_location[data_location.rfind('/', 0) + 1:]
original_data_directory = data_location[:data_location.rfind('/', 0) + 1]
print(original_data_directory)
data.to_csv((:data_location.rfind('.csv', 0)) +  '(categorized)')