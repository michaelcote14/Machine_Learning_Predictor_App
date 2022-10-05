import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

pd.options.display.width = 500
pd.set_option('display.max_columns', 500)
dataframe = pd.read_csv("data/student-mat.csv")
print('\n', dataframe.head(), '\n')


# find the unique values in each column
print('Unique Values:')
for x in dataframe.columns:
    print(x, ':', len(dataframe[x].unique()))

# gets the 10 most frequent categories
print('\n', 'Top 10 Value Counts for Mjob:\n', dataframe.Mjob.value_counts().sort_values(ascending=False).head(10))

print(dataframe['Mjob'])

ohe = OneHotEncoder()
encoded_array = ohe.fit_transform(dataframe[['Mjob']]).toarray()
print('encoded array:\n', encoded_array)

print('ohe.categories:', ohe.categories_)
feature_labels = ohe.categories_

print('np.array:\n', np.array(feature_labels).ravel())
feature_labels = np.array(feature_labels).ravel()

fully_encoded_array = pd.DataFrame(encoded_array, columns= feature_labels)
print(fully_encoded_array)

# ToDo make this variable replaceable
# ToDo this for loop friendly
# ToDo make this work.
