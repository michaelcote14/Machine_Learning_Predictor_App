import pandas as pd

original_df = pd.read_csv("../Data/nfl_data.csv")
print(original_df)
print(original_df.info())
target_variable = 'G3'

for column in original_df.columns:
    print('Column:', column)
    print('Column dtype:', original_df[column].dtypes)
    if original_df[column].dtype == 'int64':
        print('-------------working-----------------')
        original_df[column] = original_df[column].astype('int')