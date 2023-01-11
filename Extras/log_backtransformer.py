import pandas as pd
import numpy as np
from scipy.special import inv_boxcox

data = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/real_estate_data/train.csv")

print('data SalePrice:\n', data['SalePrice'].head())

# from Step_1_Data_Cleaning.skew_transformer import log_transformer
# for column in data.columns:
#     if data[column].dtypes != 'object':
#         log_skew_score, log_transformed_column = log_transformer(data, column)
#         data[column] = log_transformed_column
#
# print('data head:\n', data.head())
#
# print('Log Back-Transformed Data:\n', np.exp(data['SalePrice']))

# from Step_1_Data_Cleaning.skew_transformer import sqr_root_transformer
# for column in data.columns:
#     if data[column].dtypes != 'object':
#         skew_score, sqr_root_transformed_column = sqr_root_transformer(data, column)
#         data[column] = sqr_root_transformed_column
#
# print('data.head():\n', data.head())
#
# data[column] = data[column] ** 2
# print('final one:', data[column])

# from Step_1_Data_Cleaning.skew_transformer import box_cox_transformer
# for column in data.columns:
#     if data[column].dtypes != 'object' and float(min(data[column])) > 0:
#         box_cox_skew_score, box_cox_transformed_column, box_cox_lambda = box_cox_transformer(data, column)
#         data[column] = box_cox_transformed_column
# print(data['SalePrice'].head())
#
# print('final:\n', inv_boxcox(data['SalePrice'], box_cox_lambda))

target_variable = 'SalePrice'

from Step_1_Data_Cleaning.skew_transformer import target_transformer
from scipy.special import inv_boxcox
target_skew_winner, transformed_target_column, target_box_cox_lambda = target_transformer(data, target_variable)

back_transformed_column = inv_boxcox(transformed_target_column, target_box_cox_lambda)
print('back transformed column:\n', back_transformed_column)
