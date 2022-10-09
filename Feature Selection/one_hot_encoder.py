import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

pd.options.display.width = 500
pd.set_option('display.max_columns', 500)
original_dataframe = pd.read_csv("../Data/student-mat.csv")
print('original_dataframe.head:\n', original_dataframe.head())

all_dataframes_after_drops_single = pd.DataFrame()
for column in original_dataframe:
    print('column:\n',column)
    if original_dataframe[column].dtypes == 'object':
        try:
            series_to_encode = original_dataframe[[column]]
            print('series_to_encode:\n', series_to_encode) #happens from here

            transformer_function = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), [column]), remainder='passthrough')
            # print('transformer_function:\n', transformer_function)

            encoded_series = transformer_function.fit_transform(series_to_encode)
            # print('transformed_series\n', transformed_series)
            first_encoded_dataframe = pd.DataFrame(encoded_series, columns=transformer_function.get_feature_names_out()) #problem line for data with more than 2 answers
            print('first_encoded_dataframe:\n', first_encoded_dataframe) # to here

            # transformed_series.rename(columns={transformed_dataframe[0] : }, inplace=True) # line for changing names
            print('first_encoded_dataframe:\n', first_encoded_dataframe.head)

            column_to_drop = first_encoded_dataframe.columns[0]
            print('column to drop:\n', column_to_drop)

            single_dataframe_after_drop = first_encoded_dataframe.drop(columns=str(column_to_drop))
            print('single_dataframe_after_drop:\n', single_dataframe_after_drop)

            one_hot_default_name = single_dataframe_after_drop.columns[0]
            print('single_dataframe_after_drop.columns:\n', single_dataframe_after_drop.columns[0])
            single_dataframe_after_drop.rename(columns={single_dataframe_after_drop.columns[0] : one_hot_default_name[15: 30]}, inplace=True)
            print('single_dataframe_after_drop.columns[0]:\n', single_dataframe_after_drop.columns[0])


            all_dataframes_after_drops_single = pd.concat([all_dataframes_after_drops_single, single_dataframe_after_drop], axis=1)
            print('column:\n', column)
            print('all_dataframes_after_drops_single:\n', all_dataframes_after_drops_single)


            originaldf_column_to_drop = original_dataframe.drop(column, axis=1, inplace=True)
            print('original_dataframe.columns:\n', original_dataframe.columns)

        except Exception as e:
            print('Error:', e)
            print('got error on loop')

    else:
        print('integer\n')


print('all_dataframes_after_drops:\n', all_dataframes_after_drops_single)
final_dataframe_single_encoder = pd.concat([original_dataframe, all_dataframes_after_drops_single], axis=1)
print('final dataframe:\n', final_dataframe_single_encoder)




