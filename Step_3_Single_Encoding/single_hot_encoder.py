import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

def single_encoder(dataframe_to_encode, dataframe_to_encode2=None):
    all_dataframes_after_drops_single = pd.DataFrame()
    all_dataframes_after_drops_single2 = pd.DataFrame()
    for column in dataframe_to_encode:
        # print('column:', column, '|||')
        category_count = dataframe_to_encode[column].value_counts()

        if dataframe_to_encode[column].dtypes == 'object' and len(category_count) < 3:

            series_to_encode = dataframe_to_encode[[column]]
            series_to_encode2 = dataframe_to_encode2[[column]]
            # print('series to encode:', series_to_encode)


            encoder = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), [column]),
                                                           remainder='passthrough')

            encoded_series = encoder.fit_transform(series_to_encode)
            encoded_series2 = encoder.transform(series_to_encode2)
            first_encoded_dataframe = pd.DataFrame(encoded_series,
                                                   columns=encoder.get_feature_names_out())  # problem line for data with more than 2 answers
            first_encoded_dataframe2 = pd.DataFrame(encoded_series2, columns=encoder.get_feature_names_out())


            if len(category_count) > 1:
                column_to_drop = first_encoded_dataframe.columns[0]

                single_dataframe_after_drop = first_encoded_dataframe.drop(columns=str(column_to_drop))
                single_dataframe_after_drop2 = first_encoded_dataframe2.drop(columns=str(column_to_drop))

                one_hot_default_name = single_dataframe_after_drop.columns[0]
                single_dataframe_after_drop.rename(
                    columns={single_dataframe_after_drop.columns[0]: one_hot_default_name[15:]}, inplace=True)
                single_dataframe_after_drop2.rename(
                    columns={single_dataframe_after_drop2.columns[0]: one_hot_default_name[15:]}, inplace=True)
            else:
                one_hot_default_name = first_encoded_dataframe.columns[0]
                first_encoded_dataframe.rename(columns={first_encoded_dataframe.columns[0]: one_hot_default_name[15:]},
                                               inplace=True)
                first_encoded_dataframe2.rename(columns={first_encoded_dataframe2.columns[0]: one_hot_default_name[15:]},
                                                inplace=True)
                single_dataframe_after_drop = first_encoded_dataframe
                single_dataframe_after_drop2 = first_encoded_dataframe2

            all_dataframes_after_drops_single = pd.concat(
                [all_dataframes_after_drops_single, single_dataframe_after_drop], axis=1)
            all_dataframes_after_drops_single2 = pd.concat(
                [all_dataframes_after_drops_single2, single_dataframe_after_drop2], axis=1)

            original_df_column_to_drop = dataframe_to_encode.drop(column, axis=1, inplace=True)
            original_df_column_to_drop2 = dataframe_to_encode2.drop(column, axis=1, inplace=True)


    final_dataframe_single_encoder = pd.concat([dataframe_to_encode, all_dataframes_after_drops_single], axis=1)
    final_dataframe_single_encoder2 = pd.concat([dataframe_to_encode2, all_dataframes_after_drops_single2], axis=1)

    return all_dataframes_after_drops_single, all_dataframes_after_drops_single2

# def single_encoder_for_test_df(dataframe_to_encode, encoder):
#     for column in dataframe_to_encode:
#         print('\n\ncolumn:', column, '|||')
#         category_count = dataframe_to_encode[column].value_counts()
#
#         if dataframe_to_encode[column].dtypes == 'object' and len(category_count) < 3:
#
#             series_to_encode = dataframe_to_encode[[column]]
#             print('series to encode:', series_to_encode)
#
#             # encoded_series = encoder.transform(series_to_encode)
#             #
#             # first_encoded_dataframe = pd.DataFrame(encoded_series,
#             #                                        columns=encoder.get_feature_names_out())  # problem line for data with more than 2 answers
#             #
#             # if len(category_count) > 1:
#             #     column_to_drop = first_encoded_dataframe.columns[0]
#             #
#             #     single_dataframe_after_drop = first_encoded_dataframe.drop(columns=str(column_to_drop))
#             #
#             #     one_hot_default_name = single_dataframe_after_drop.columns[0]
#             #     single_dataframe_after_drop.rename(
#             #         columns={single_dataframe_after_drop.columns[0]: one_hot_default_name[15:]}, inplace=True)
#             # else:
#             #     one_hot_default_name = first_encoded_dataframe.columns[0]
#             #     first_encoded_dataframe.rename(columns={first_encoded_dataframe.columns[0]: one_hot_default_name[15:]},
#             #                                    inplace=True)
#             #     single_dataframe_after_drop = first_encoded_dataframe
#             #
#             # single_encoded_df = pd.concat(
#             #     [single_encoded_df, single_dataframe_after_drop], axis=1)
#
#         #     original_df_column_to_drop = dataframe_to_encode.drop(column, axis=1, inplace=True)
#         #
#         # final_dataframe_single_encoder = pd.concat([dataframe_to_encode, single_encoded_df], axis=1)
#
#     # return single_encoded_df







