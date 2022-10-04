import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

pd.options.display.width = 500
pd.set_option('display.max_columns', 500)
dataframe = pd.read_csv("data/student-mat.csv")
print(dataframe.head)

# gets the data for only categorical series
new_dataframe = dataframe.select_dtypes(include='O')
print(new_dataframe)

# prints the number of unique categories for each series
for x in new_dataframe.columns:
    print(x, ':', len(new_dataframe[x].unique()))

# finds the top 20 most common categories for each series
print(new_dataframe.Mjob.value_counts().sort_values(ascending=False).head(20))

# make list with top 10 variables
top_10 = [x for x in new_dataframe.value_counts().sort_values(ascending=False).head(10).index]
print('top_10:\n', top_10)

# make the top 10 variables binary
for label in top_10:
    new_dataframe[label] = np.where(new_dataframe['Mjob']==label,1,0)
    # new_dataframe[['Mjob'] +top_10] # problem
    print(new_dataframe)


# ToDo make this work.
