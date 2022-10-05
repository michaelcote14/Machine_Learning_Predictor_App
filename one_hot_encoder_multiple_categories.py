import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

from one_hot_encoder import * # this is running the one_hot_encoder program

pd.options.display.width = 500
pd.set_option('display.max_columns', 500)
original_dataframe = pd.read_csv("data/student-mat.csv")

# drop the encoded dataframes from the original dataframe # maybe use merge?
# step one: get both programs' encoded dataframes together-done
# step two: combine two_program dataframe with original dataframe without the columns that were encoded
    # step 2a: drop the columns that are getting encoded from the original dataframe
        # step 2aa: drop the columns of the multiple category encoder
        # step 2ab: drop the columns of the single one hot encoder


all_dataframes_after_drops = pd.DataFrame()
for column in original_dataframe.columns:
    unique_value_amount = len(original_dataframe[column].unique())
    print('\nColumn:', column)
    print('Unique Value Amount:', unique_value_amount)


    if unique_value_amount > 2 and original_dataframe[column].dtypes == 'object':

        # spot to drop these columns from the original dataframe
        original_dataframe.drop([column], axis=1, inplace=True)
        print('original dataframe right after if block:\n', original_dataframe.head)
        # this ruins the below pieces

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


    else:
        print('did not work on this column')
        # this is where i need to drop the columns from the original dataframe #have not tested the below yet
        print('pre-original_dataframe(else)\n', original_dataframe)
        original_dataframe.drop([column], axis=1, inplace=True)
        print('original_dataframe(else):\n', original_dataframe)
        continue


print(all_dataframes_after_drops_single)

# dataframe concatenator:
print('original_dataframe:\n', original_dataframe)
print('All Dataframes After Drops:\n', all_dataframes_after_drops)
#need this guy^ combined with one_hot_encoder's all_dataframes_after_drops_single
two_program_dataframe = pd.concat([all_dataframes_after_drops, all_dataframes_after_drops_single], axis=1)
print('two_program_dataframe:\n', two_program_dataframe)

end_dataframe = pd.concat([two_program_dataframe, original_dataframe], axis=1)
print('end:\n',end_dataframe)
# make original have G1, G2, G3 and all numerical series
print('original:\n', original_dataframe)










#
# final_dataframe = pd.concat([original_dataframe, two_dropped_dataframes], axis=1)
# print('final dataframe:\n', final_dataframe)
#
# df2 = final_dataframe.columns.drop_duplicates()
# print(final_dataframe)
# print('duplicated\n', final_dataframe.duplicated())
#
# final_dataframe = final_dataframe.loc[:, ~final_dataframe.columns.duplicated()].copy()
# print(final_dataframe)
#
# final_dataframe.drop('Unnamed: 33', axis=1, inplace=True)
#
# final_dataframe = final_dataframe.sort_index(axis=1)
# print(final_dataframe)

# ToDo some dataframes still need to be dropped, Fjob, Mjob, activities, address, famsize, famsup, higher, nursery
# this is because the other dataframes are bringing in the old dataframes

