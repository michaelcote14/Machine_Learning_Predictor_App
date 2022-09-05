import pandas as pd #this is to read in data sheets
import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot #this allows you to make graphs
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
from matplotlib import style #this changes the style of your plot's grid
import seaborn as seabornInstance
from sklearn.linear_model import LinearRegression
from sklearn import metrics



dataset = pd.read_csv(r'C:\Users\micha\PycharmProjects\tensorEnv\student-mat.csv', sep=';')
print(dataset.shape) #shows how many rows and columns you have in your csv file

print(dataset.describe()) #shows some of your columns and rows

dataset.plot(x='G1', y='G3', style='o') #This whole section gives a scatter plot between two columns to look for corellation
pyplot.title('G1 vs G3')
pyplot.xlabel('G1')
pyplot.ylabel('G3')
#pyplot.show()

pyplot.figure(figsize=(15,10)) #this section presents the mean of one variable in a bar graph
pyplot.tight_layout()
seabornInstance.distplot(dataset['G3'])
#pyplot.show()


X = dataset['G1'].values.reshape(-1,1) #these two lines just extract the data from the csv and assigns X and y to a certain column
y = dataset['G3'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #random state sets the random generator, if you use the same random state, your data will always be randomized the same. This allows you to keep the numbers.

regressor = LinearRegression()
regressor.fit(X_train, y_train) #this trains the algorithm

print('Intercept:', regressor.intercept_) #this reveals the regression line's intercept
print('Coefficient', regressor.coef_) #this reveals the regression line's coefficient, so this pretty much shows the slope and significant your x variable is when it comes to valueing the y variable. Look below for more info.
# so if the coefficient is 1.06, this means that for every 1 unit the x increases, your y will increase by 1.06. Less than 1 means there is a negative correlation, and more than 1 means a positive one.

#this section sees how accurate your current algorithm is
y_pred = regressor.predict(X_test) #this passes your testing data in
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)


#this section makes a bar graph that compares the predicted to the actual data.
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
pyplot.grid(which='major', linestyle='-', linewidth='0.5', color='green')
pyplot.grid(which='minor', linestyle=':', linewidth=' 0.5', color='black')
#pyplot.show()

#this section makes a graph that shows the regression line that the algorithm came up with to make predictions. This is not super accurate, so don't rely on it too much
pyplot.scatter(X_test, y_test, color='gray')
pyplot.plot(X_test, y_pred, color='red', linewidth=2)
#pyplot.show()


#this section shows how well a prediction algorithm performed on a data set
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) #this shows the average error size
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))