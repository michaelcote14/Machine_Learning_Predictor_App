import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

pd.options.display.width = 500
pd.set_option('display.max_columns', 500)
data = pd.read_csv("data/student-mat.csv")

all_after_drop_dataframes = pd.DataFrame()

appended_data_frame = pd.DataFrame()
print('empty data frame:\n', appended_data_frame)
for column in data:
    print('column:\n',column)
    if data[column].dtypes == 'object':
        try:
            print('not an integer\n')

            series_to_transform = data[[column]]
            print('series_to_transform:\n', series_to_transform)

            transformer_function = make_column_transformer((OneHotEncoder(), [column]), remainder='passthrough')
            print('transformer:\n', transformer_function)

            transformed_series = transformer_function.fit_transform(series_to_transform)
            transformed_dataframe = pd.DataFrame(transformed_series, columns=transformer_function.get_feature_names_out()) #problem line for data with more than 2 answers
            # transformed_df.rename(columns={''})
            print('transformed_df.head:\n', transformed_dataframe.head)

            column_to_drop = transformed_dataframe.columns[0]
            print('column to drop:\n', column_to_drop)

            dataframe_after_drop = transformed_dataframe.drop(columns=str(column_to_drop))
            print('series after drop:\n', dataframe_after_drop)


            all_after_drop_dataframes = pd.concat([all_after_drop_dataframes, dataframe_after_drop], axis=1)

            originaldf_column_to_drop = data.drop(columns=str(column))
            print('originaldf_column_to_drop:\n', originaldf_column_to_drop)
            #ToDo fix the above

            # ToDo drop the original columns that aren't yet onehot encoded




        except Exception as e:
            print('Error:', e)
            print('got error on loop')






    else:
        print('integer\n')


# print('final dataframe:\n', final_data_frame)
print('all after drop dataframes:\n', all_after_drop_dataframes)
final_dataframe = pd.concat([data, all_after_drop_dataframes], axis=1)
print('final dataframe:\n', final_dataframe)




