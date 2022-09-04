import pandas as pd

df = pd.read_csv("student-mat.csv", sep=";")
print(df)

X = df[['G1', 'G2', 'absences', 'age']] #puts all this data for these columns in X for your independant variables
y = df[['G3']] #puts your dependant variable into y

from sklearn import linear_model


