


def multiple_encoder():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    import single_hot_encoder

    single_encoded_df = single_hot_encoder.single_encoder()
    pd.options.display.width = 500
    pd.set_option('display.max_columns', 500)
    original_dataframe = pd.read_csv("Data/student-mat.csv")

    all_dataframes_after_drops = pd.DataFrame()
    for column in original_dataframe.columns:
        unique_value_amount = len(original_dataframe[column].unique())


        if unique_value_amount > 2 and original_dataframe[column].dtypes == 'object':

            ohe = OneHotEncoder(handle_unknown='ignore')

            series_to_encode = ohe.fit_transform(original_dataframe[[column]]).toarray()

            categories_getting_encoded = ohe.categories_

            categories_getting_encoded = np.array(categories_getting_encoded).ravel()

            encoded_series = pd.DataFrame(series_to_encode, columns= categories_getting_encoded)

            column_to_drop = encoded_series.columns[0]

            single_dataframe_after_drop = encoded_series.drop(columns=(str(column_to_drop)))

            new_single_dataframe_after_drop = single_dataframe_after_drop.add_prefix(column + '_')


            all_dataframes_after_drops = pd.concat([all_dataframes_after_drops, new_single_dataframe_after_drop], axis=1)

            original_dataframe.drop(column, axis=1, inplace=True)


        else:
            if original_dataframe[column].dtypes == 'object':
                original_dataframe.drop(column, axis=1, inplace=True)
            continue





    end_dataframe = pd.concat([all_dataframes_after_drops, original_dataframe], axis=1)


    encoded_df = pd.concat([end_dataframe, single_encoded_df], axis=1)

    encoded_df.sort_index(axis=1, inplace=True)

    for column in encoded_df.columns:
        if column.startswith('Unnamed'):
            encoded_df.drop([column], axis=1, inplace=True)
    return encoded_df

def multiple_encoder_printer():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    import single_hot_encoder

    single_encoded_df = single_hot_encoder.single_encoder()
    pd.options.display.width = 500
    pd.set_option('display.max_columns', 500)
    original_dataframe = pd.read_csv("Data/student-mat.csv")

    all_dataframes_after_drops = pd.DataFrame()
    for column in original_dataframe.columns:
        unique_value_amount = len(original_dataframe[column].unique())
        print('\nColumn:', column)
        print('Unique Value Amount:', unique_value_amount)


        if unique_value_amount > 2 and original_dataframe[column].dtypes == 'object':

            ohe = OneHotEncoder(handle_unknown='ignore')

            series_to_encode = ohe.fit_transform(original_dataframe[[column]]).toarray()
            print('series_to_encode:\n', series_to_encode)

            categories_getting_encoded = ohe.categories_

            categories_getting_encoded = np.array(categories_getting_encoded).ravel()
            print('columns getting encoded:\n', np.array(categories_getting_encoded).ravel())

            encoded_series = pd.DataFrame(series_to_encode, columns= categories_getting_encoded)
            print('encoded_series:\n', encoded_series)

            column_to_drop = encoded_series.columns[0]
            print('column_to_drop:\n', column_to_drop)

            single_dataframe_after_drop = encoded_series.drop(columns=(str(column_to_drop)))
            print('single_dataframe_after_drop:\n', single_dataframe_after_drop)

            new_single_dataframe_after_drop = single_dataframe_after_drop.add_prefix(column + '_')
            print('single_dataframe_after_drop*renamed:\n', single_dataframe_after_drop)


            all_dataframes_after_drops = pd.concat([all_dataframes_after_drops, new_single_dataframe_after_drop], axis=1)
            print('all_dataframes_after_drops\n', all_dataframes_after_drops)

            original_dataframe.drop(column, axis=1, inplace=True)


        else:
            print('did not work on this column')
            if original_dataframe[column].dtypes == 'object':
                original_dataframe.drop(column, axis=1, inplace=True)
            continue




    print('original_dataframe:\n', original_dataframe)
    print('All Dataframes After Drops:\n', all_dataframes_after_drops)

    end_dataframe = pd.concat([all_dataframes_after_drops, original_dataframe], axis=1)
    print('end:\n',end_dataframe)


    encoded_df = pd.concat([end_dataframe, single_encoded_df], axis=1)
    print('last_dataframe1:\n', encoded_df)

    encoded_df.sort_index(axis=1, inplace=True)
    print('last_dataframe2:\n', encoded_df)

    for column in encoded_df.columns:
        if column.startswith('Unnamed'):
            encoded_df.drop([column], axis=1, inplace=True)
    print('last:\n', encoded_df)
    return encoded_df


if __name__ == '__main__':
    multiple_encoder()











