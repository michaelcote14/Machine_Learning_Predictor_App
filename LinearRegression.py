import pandas as pd

df = pd.read_csv("student-mat.csv", sep=";")
print(df)

X = df[['G1', 'G2', 'absences', 'age']] #puts all this data for these columns in X for your independant variables
y = df[['G3']] #puts your dependant variable into y

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedG3 = regr.predict([[11, 12, 2, 20]]) #this will predict a G3 based on the inputed G1, G2, absences, and age
print(predictedG3)

print(regr.coef_) #this tells the coefficients for each input, aka the impact they have
