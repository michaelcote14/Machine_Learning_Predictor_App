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
            print('series_to_transform:\n', series_to_transform) #happens from here

            transformer_function = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), [column]), remainder='passthrough')
            # print('transformer_function:\n', transformer_function)

            transformed_series = transformer_function.fit_transform(series_to_transform)
            # print('transformed_series\n', transformed_series)
            transformed_dataframe = pd.DataFrame(transformed_series, columns=transformer_function.get_feature_names_out()) #problem line for data with more than 2 answers
            print('transformed_dataframe\n', transformed_dataframe) # to here

            # transformed_series.rename(columns={transformed_dataframe[0] : }, inplace=True) # line for changing names
            print('transformed_df.head:\n', transformed_dataframe.head)

            column_to_drop = transformed_dataframe.columns[0]
            print('column to drop:\n', column_to_drop)

            dataframe_after_drop = transformed_dataframe.drop(columns=str(column_to_drop))
            print('series after drop:\n', dataframe_after_drop)

            one_hot_default_name = dataframe_after_drop.columns[0]
            print('dataframe_after_drop.columns:\n', dataframe_after_drop.columns[0])
            dataframe_after_drop.rename(columns={dataframe_after_drop.columns[0] : one_hot_default_name[15: 30]}, inplace=True)
            print('dataframe_after_drop.columns[0]:\n', dataframe_after_drop.columns[0])


            all_after_drop_dataframes = pd.concat([all_after_drop_dataframes, dataframe_after_drop], axis=1)
            print('column:\n', column)



            originaldf_column_to_drop = data.drop(column, axis=1, inplace=True)
            print('originaldf_column_to_drop:\n', originaldf_column_to_drop)


            # ToDo adapt it to using multiple categories




        except Exception as e:
            print('Error:', e)
            print('got error on loop')






    else:
        print('integer\n')


# print('final dataframe:\n', final_data_frame)
print('all after drop dataframes:\n', all_after_drop_dataframes)
final_dataframe = pd.concat([data, all_after_drop_dataframes], axis=1)
print('final dataframe:\n', final_dataframe)




