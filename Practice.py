import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

pd.options.display.width = 500
pd.set_option('display.max_columns', 500)
data = pd.read_csv("data/student-mat.csv")


for column in data:
    print('column:\n',column)
    if data[column].dtypes == 'object':
        try:
            print('not an integer\n')

            series_to_transform = data[[column]]
            print('chosen_data:\n', series_to_transform)

            transformer_function = make_column_transformer((OneHotEncoder(), [column]), remainder='passthrough')
            print('transformer:\n', transformer_function)

            transformed_series = transformer_function.fit_transform(series_to_transform)
            transformed_dataframe = pd.DataFrame(transformed_series, columns=transformer_function.get_feature_names_out()) #problem line for data with more than 2 answers
            # transformed_df.rename(columns={''})
            print('transformed_df.head:\n', transformed_dataframe.head)

            column_to_drop = transformed_dataframe.columns[0]
            print('column to drop:\n', column_to_drop)

            series_after_drop = transformed_dataframe.drop(columns=str(column_to_drop))
            print('series after drop:\n', series_after_drop)

            #ToDo append the above dataframe with the newly made dataframe each time




        except Exception as e:
            print('Error:', e)
            print('got error on loop')






    else:
        print('integer\n')


