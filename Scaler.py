import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pd.read_csv("student-mat.csv", sep=";")

X = df[['G1', 'G2', 'absences', 'age']] #puts all this data for these columns in X for your independant variables
y = df[['G3']]

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[5, 6, 3, 20]])

predictedG3 = regr.predict([scaled[0]])
print('predictedG3:',predictedG3)