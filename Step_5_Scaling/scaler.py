import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Extras.functions import time_formatter
import time



def get_predictor_array(self, data_we_know_dict, fully_cleaned_df, target_variable):
    data_we_have_dataframe = pd.DataFrame.from_dict(data_we_know_dict)
    # combines the two dataframes and gives null values the mean
    new_dataframe = pd.concat([data_we_have_dataframe, fully_cleaned_df])
    new_dataframe.fillna(fully_cleaned_df.mean())
    # separates the two dataframes again
    new_dataframe = new_dataframe.iloc[0]
    new_dataframe = pd.DataFrame(new_dataframe)
    new_dataframe = new_dataframe.T
    new_dataframe = new_dataframe.drop([target_variable], axis=1)
    unscaled_predictor_array = new_dataframe.fillna(fully_cleaned_df.mean())
    unscaled_predictor_array = np.array(unscaled_predictor_array)
    return unscaled_predictor_array  # not in order
