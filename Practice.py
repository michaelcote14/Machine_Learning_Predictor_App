import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

pd.options.display.width = 500
pd.set_option('display.max_columns', 500)
dataframe = pd.read_csv("data/student-mat.csv")
print('\n', dataframe.head(), '\n')


for column in dataframe.columns:
    unique_value_amount = len(dataframe[column].unique())
    print('\nColumn:', column)
    print('Unique Value Amount:', unique_value_amount)
    if unique_value_amount > 2 and dataframe[column].dtypes == 'object':

        ohe = OneHotEncoder()

        first_encoded_array = ohe.fit_transform(dataframe[[column]]).toarray()
        print('encoded array:\n', first_encoded_array)

        columns_getting_encoded = ohe.categories_

        columns_getting_encoded = np.array(columns_getting_encoded).ravel()
        print('columns getting encoded:\n', np.array(columns_getting_encoded).ravel())

        fully_encoded_array = pd.DataFrame(first_encoded_array, columns= columns_getting_encoded)
        print(fully_encoded_array)

        # ToDo combine each loop's array into one massive array

    else:
        print('did not work on this column')
        continue


# for column in dataframe:
#     if dataframe[column].dtypes =