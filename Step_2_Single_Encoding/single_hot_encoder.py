import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from Step_1_Visualizing.visualization import type_clean_df


def single_encoder_printer():
        pd.options.display.width = 500
        pd.set_option('display.max_columns', 500)
        print('type_clean_df.head:\n', type_clean_df.head())

        all_dataframes_after_drops_single = pd.DataFrame()
        for column in type_clean_df:
            print('\ncolumn:',column)
            category_count = type_clean_df[column].value_counts()
            print('Length of Unique Values:', len(category_count))
            print('Dtypes:', type_clean_df[column].dtypes)


            if type_clean_df[column].dtypes == 'object' and len(category_count) < 3:

                series_to_encode = type_clean_df[[column]]
                print('series_to_encode:\n', series_to_encode) #happens from here

                transformer_function = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), [column]), remainder='passthrough')
                # print('transformer_function:\n', transformer_function)

                encoded_series = transformer_function.fit_transform(series_to_encode)
                # print('transformed_series\n', transformed_series)
                first_encoded_dataframe = pd.DataFrame(encoded_series, columns=transformer_function.get_feature_names_out()) #problem line for data with more than 2 answers
                print('first_encoded_dataframe:\n', first_encoded_dataframe) # to here

                # transformed_series.rename(columns={transformed_dataframe[0] : }, inplace=True) # line for changing names
                print('first_encoded_dataframe:\n', first_encoded_dataframe.head)

                if len(category_count) > 1:
                    column_to_drop = first_encoded_dataframe.columns[0]
                    print('column to drop:\n', column_to_drop)

                    single_dataframe_after_drop = first_encoded_dataframe.drop(columns=str(column_to_drop))
                    print('single_dataframe_after_drop:\n', single_dataframe_after_drop)

                    one_hot_default_name = single_dataframe_after_drop.columns[0]
                    print('single_dataframe_after_drop.columns:\n', single_dataframe_after_drop.columns[0])
                    single_dataframe_after_drop.rename(columns={single_dataframe_after_drop.columns[0] : one_hot_default_name[15:]}, inplace=True)
                    print('single_dataframe_after_drop.columns[0]:\n', single_dataframe_after_drop.columns[0])
                else:
                    print('Test-----:', first_encoded_dataframe.columns[0])
                    one_hot_default_name = first_encoded_dataframe.columns[0]
                    first_encoded_dataframe.rename(columns={first_encoded_dataframe.columns[0]: one_hot_default_name[15: ]}, inplace=True)
                    single_dataframe_after_drop = first_encoded_dataframe
                    print('else is activated')


                all_dataframes_after_drops_single = pd.concat([all_dataframes_after_drops_single, single_dataframe_after_drop], axis=1)
                print('column:\n', column)
                print('all_dataframes_after_drops_single:\n', all_dataframes_after_drops_single)


                originaldf_column_to_drop = type_clean_df.drop(column, axis=1, inplace=True)
                print('type_clean_df.columns:\n', type_clean_df.columns)

        else:
            print('integer\n')


        print('all_dataframes_after_drops_single:\n', all_dataframes_after_drops_single)
        final_dataframe_single_encoder = pd.concat([type_clean_df, all_dataframes_after_drops_single], axis=1)
        print('Single Encoded Dataframe:\n', final_dataframe_single_encoder)
        print('All Dataframes Returner:\n', all_dataframes_after_drops_single)
        return all_dataframes_after_drops_single


def single_encoder():
    all_dataframes_after_drops_single = pd.DataFrame()
    for column in type_clean_df:
        category_count = type_clean_df[column].value_counts()

        if type_clean_df[column].dtypes == 'object' and len(category_count) < 3:

            series_to_encode = type_clean_df[[column]]

            transformer_function = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), [column]),
                                                           remainder='passthrough')
            # print('transformer_function:\n', transformer_function)

            encoded_series = transformer_function.fit_transform(series_to_encode)
            # print('transformed_series\n', transformed_series)
            first_encoded_dataframe = pd.DataFrame(encoded_series,
                                                   columns=transformer_function.get_feature_names_out())  # problem line for data with more than 2 answers

            # transformed_series.rename(columns={transformed_dataframe[0] : }, inplace=True) # line for changing names

            if len(category_count) > 1:
                column_to_drop = first_encoded_dataframe.columns[0]

                single_dataframe_after_drop = first_encoded_dataframe.drop(columns=str(column_to_drop))

                one_hot_default_name = single_dataframe_after_drop.columns[0]
                single_dataframe_after_drop.rename(
                    columns={single_dataframe_after_drop.columns[0]: one_hot_default_name[15:]}, inplace=True)
            else:
                one_hot_default_name = first_encoded_dataframe.columns[0]
                first_encoded_dataframe.rename(columns={first_encoded_dataframe.columns[0]: one_hot_default_name[15:]},
                                               inplace=True)
                single_dataframe_after_drop = first_encoded_dataframe

            all_dataframes_after_drops_single = pd.concat(
                [all_dataframes_after_drops_single, single_dataframe_after_drop], axis=1)

            originaldf_column_to_drop = type_clean_df.drop(column, axis=1, inplace=True)


    final_dataframe_single_encoder = pd.concat([type_clean_df, all_dataframes_after_drops_single], axis=1)
    return all_dataframes_after_drops_single


single_encoded_df = single_encoder()
print('Single Encoded Df:\n', single_encoded_df)

if __name__ == '__main__':
    pass
    # all_dataframes_after_drops_single = single_encoder_printer()
    # print('Single Encoder Printer():\n', all_dataframes_after_drops_single)




