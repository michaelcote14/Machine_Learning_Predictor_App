import pandas as pd

df = pd.read_csv("student-mat.csv", sep=";")
print(df)

X = df[['G1', 'G2', 'absences', 'age']] #puts all this data for these columns in X for your independant variables
y = df[['G3']] #puts your dependant variable into y

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y) #creates the regression line

predictedG3 = regr.predict([[5, 6, 3, 20]]) #this will predict a G3 based on the inputed G1, G2, absences, and age
print('predictedG3:', predictedG3)

#this section scales the data to be more equal
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scaledX = scale.fit_transform(X)
print(scaledX)


print('new') #just so I can see where the new line is
print(regr.coef_) #this tells the coefficients for each input, aka the impact they have
