import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

pd.set_option('display.max_columns', None)
data = pd.read_csv("data/student-mat.csv")
chosen_data = data[['sex']]

transformer = make_column_transformer((OneHotEncoder(), ['sex']), remainder='passthrough')

transformed = transformer.fit_transform(chosen_data)
transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

pd.set_option('display.max_columns', None)

print(transformed_df.head())

merged = data.merge(chosen_data)
print('merged:', merged.head)




#ToDo take out the unnecessary columns
#ToDo merge the two data frames







