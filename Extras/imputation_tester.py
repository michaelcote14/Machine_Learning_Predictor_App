from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np

dataframe = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/nfl_data(numeric_only).csv")
dataframe = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/nfl_data.csv")

numerical_dataframe = dataframe.select_dtypes(['number'])
print('numerical dataframe:\n', numerical_dataframe.head())

categorical_dataframe = dataframe.select_dtypes(include='object')
print('categorical dataframe:\n', categorical_dataframe.head())

# Impute the numerical dataframe only
print('Null Value Count:\n', numerical_dataframe.isnull().sum())

imputer = IterativeImputer()
imputed_array = imputer.fit_transform(numerical_dataframe)
imputed_df = pd.DataFrame(imputed_array, columns=numerical_dataframe.columns)
print('imputed df:\n', imputed_df)
print(imputed_df.isnull().sum())

# Add numeric columns back to original dataframe
fully_imputed_df = pd.concat([categorical_dataframe, imputed_df], axis=1)
print('fully imputed df:\n', fully_imputed_df)