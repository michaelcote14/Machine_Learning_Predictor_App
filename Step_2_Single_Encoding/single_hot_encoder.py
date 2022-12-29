import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

def single_encoder(type_clean_df):
    all_dataframes_after_drops_single = pd.DataFrame()
    for column in type_clean_df:
        category_count = type_clean_df[column].value_counts()

        if type_clean_df[column].dtypes == 'object' and len(category_count) < 3:

            series_to_encode = type_clean_df[[column]]

            transformer_function = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), [column]),
                                                           remainder='passthrough')

            encoded_series = transformer_function.fit_transform(series_to_encode)
            first_encoded_dataframe = pd.DataFrame(encoded_series,
                                                   columns=transformer_function.get_feature_names_out())  # problem line for data with more than 2 answers


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



if __name__ == '__main__':
    pass
    # all_dataframes_after_drops_single = single_encoder_printer()
    # print('Single Encoder Printer():\n', all_dataframes_after_drops_single)




