import random
import pandas_tutorital as pd #this is to read in data sheets
import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot #this allows you to make graphs
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
from matplotlib import style #this changes the style of your plot's grid

UploadFile = "student-mat(Numerical Only).csv"
Dataframe = pd.read_csv(UploadFile, sep=',')
print("Dataframe Columns:", Dataframe.columns)
DataColumns = Dataframe.columns
DataFeatures = DataColumns.drop("G3")
print("Data Columns:", DataColumns)
print("DataFeatures:", DataFeatures)



for i in range(50):
    SampledNumber = random.sample(range(3), 1) #((random range of numbers)), how many numbers in list)
    TestList = [DataFeatures]
    SampledList = random.sample(DataFeatures, SampledNumber[0]+1)

    print(SampledNumber)
    print(SampledList)

# sampled_list = random.sample(TestList, random.sample(range(5), 1))
# print(sampled_list)

CombinationCount = 0
# for k in TestList:
#     print("k:", k)
#     for j in TestList:
#         print("j:", j)
#         if j != k:
#             print("working")
#             print("k, j:", k,j)
#         CombinationCount = CombinationCount + 1
#         print("Combination Count:",CombinationCount)
