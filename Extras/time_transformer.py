import pandas as pd
import numpy as np

dataframe = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/real_estate_data/train.csv")

print(dataframe['MoSold'].unique())

# How to transform time based on months
dataframe['MoSold'] = np.cos(0.5236 * dataframe['MoSold'])

print(dataframe['MoSold'])