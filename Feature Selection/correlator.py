import pandas as pd
import sklearn
from one_hot_encoder_multiple_categories import last_dataframe
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataframe = last_dataframe

target_name = 'G3'

y = dataframe[target_name]
X = dataframe.drop(target_name, axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

plt.figure(figsize=(200, 100))
cor = X_train.corr()
g = sns.heatmap(cor, cmap=plt.cm.bone,  annot=True)
g.set_yticklabels(g.get_yticklabels(), rotation=00)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.show()

def correlation(dataframe, threshold):
    col_corr = set()
    corr_matrix = dataframe.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.4)
len(set(corr_features))
print('correlated features:', len(set(corr_features)))
print('correlated features are:', corr_features)