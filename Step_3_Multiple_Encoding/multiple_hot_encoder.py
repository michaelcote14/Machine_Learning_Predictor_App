import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def multiple_encoder(original_df, single_encoded_df):
    pd.options.display.width = 500
    pd.set_option('display.max_columns', 500)

    all_dataframes_after_drops = pd.DataFrame()
    for column in original_df.columns:
        unique_value_amount = len(original_df[column].unique())


        if unique_value_amount > 2 and original_df[column].dtypes == 'object':
            ohe = OneHotEncoder(handle_unknown='ignore')

            series_to_encode = ohe.fit_transform(original_df[[column]]).toarray()

            categories_getting_encoded = ohe.categories_

            categories_getting_encoded = np.array(categories_getting_encoded).ravel()

            encoded_series = pd.DataFrame(series_to_encode, columns= categories_getting_encoded)

            column_to_drop = encoded_series.columns[0]

            single_dataframe_after_drop = encoded_series.drop(columns=(str(column_to_drop)))

            new_single_dataframe_after_drop = single_dataframe_after_drop.add_prefix(column + '_')


            all_dataframes_after_drops = pd.concat([all_dataframes_after_drops, new_single_dataframe_after_drop], axis=1)

            original_df.drop(column, axis=1, inplace=True)


        else:
            if original_df[column].dtypes == 'object':
                original_df.drop(column, axis=1, inplace=True)
            continue





    end_dataframe = pd.concat([all_dataframes_after_drops, original_df], axis=1)


    encoded_df = pd.concat([end_dataframe, single_encoded_df], axis=1)

    encoded_df.sort_index(axis=1, inplace=True)

    return encoded_df














