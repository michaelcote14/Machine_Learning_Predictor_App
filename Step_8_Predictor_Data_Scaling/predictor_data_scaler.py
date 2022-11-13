import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from Step_3_Multiple_Encoding.multiple_hot_encoder import multiple_encoded_df
from Step_1_Visualizing.visualization import target_variable
from Step_5_Scaling.scaler import main_scaler
predictor_data_dict = {'age': [16.0], 'G2': [10.0], 'goout': [3], 'internet_yes': [1]}
start_time = time.time()
FEATURES = multiple_encoded_df.drop([target_variable], axis=1)
X = np.array(FEATURES)
y = np.array(multiple_encoded_df[target_variable])
RUNTIMES = 100


def get_predictor_array(predictor_data_dict):
    data_we_have_dataframe = pd.DataFrame.from_dict(predictor_data_dict)
    # combines the two dataframes and gives null values the mean
    new_dataframe = pd.concat([data_we_have_dataframe, multiple_encoded_df])
    new_dataframe.fillna(multiple_encoded_df.mean())
    # separates the two dataframes again
    new_dataframe = new_dataframe.iloc[0]
    new_dataframe = pd.DataFrame(new_dataframe)
    new_dataframe = new_dataframe.T
    new_dataframe = new_dataframe.drop([target_variable], axis=1)
    unscaled_predictor_array = new_dataframe.fillna(multiple_encoded_df.mean())
    unscaled_predictor_array = np.array(unscaled_predictor_array)
    return unscaled_predictor_array
print(get_predictor_array(predictor_data_dict))

def predictor_data_scaler(predictor_data_dict):
    unscaled_predictor_array = get_predictor_array(predictor_data_dict)
    scaled_df, scaled_predictor_array = main_scaler(unscaled_predictor_array)
    scaled_predictor_df = pd.DataFrame(scaled_predictor_array, columns=FEATURES.columns)

    # this creates the dataframe with scaled data from input data only
    key_lst = list(predictor_data_dict.keys())
    value_lst = list(predictor_data_dict.values())
    scaled_predictor_df = scaled_predictor_df.loc[:, key_lst]
    scaled_predictor_df.to_csv('scaled_predictor_df.csv', index=False, encoding='utf-8')
    return scaled_predictor_array, scaled_predictor_df

