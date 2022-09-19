import pandas as pd #this is to read in data sheets
import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
import itertools
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot #this allows you to make graphs
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
from matplotlib import style #this changes the style of your plot's grid


Dataframe = pd.read_csv("Feature Optimizer Data.csv", sep=',')
print("Dataframe Columns:", Dataframe.columns)
DataColumns = Dataframe.columns
DataFeatures = DataColumns.drop("G3")
DataFeaturesList = DataFeatures.tolist()
print(DataFeaturesList)
print("DataFeatures:", DataFeatures)





combinations = 0
for i in DataFeaturesList:
    result = itertools.combinations(DataFeaturesList, DataFeaturesList.index(i)+1)
    print("i:", i)
    for item in result:
        print(item)
        combinations = combinations + 1
print("combinations:", combinations)





