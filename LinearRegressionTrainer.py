import pandas as pd #this is to read in data sheets
import numpy as np #this is for doing interesting things wtih numbers
import sklearn #this is the machine learning module
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot #this allows you to make graphs
import pickle #this saves your model for the machine and keeps you from having to retrain plus it saves your most accurate model
from matplotlib import style #this changes the style of your plot's grid

data = pd.read_csv("Data/student-mat.csv", sep=",") #reads in the data and has to separate them using ;

data = data[["Medu", "Fedu", "G1", "G2", "studytime", "famrel", "freetime", "traveltime", "failures", "health", "Walc", "Dalc", "G3"]] #these are the attributes we want to analyze, you can have one of these or a ton of them

predict = "G3" #this is the target label we are trying to predict using the data
#print(data) #shows all your data for the columns you want

X = np.array(data.drop(["G3"], axis=1)) #this gets rid of the array of all G3 data and keeps it from being used to as a feature/attribute
y = np.array(data["G3"]) #this creates an array in your output that is for G3
#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) #this puts the data into 4 different arrays: x train, x test, y train, and y test

Runtimes = 100000
PickleBest, best, TotalAccuracy = 0, 0, 0 # this sets a baseline accuracy prediction so that the machine has something to outperform       #comment the following marked positions when you want to stop training your data
for _ in range(Runtimes): #runs the model 30 times  #this also gives us 30 new data sets each time                  #
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)    #this randomly places all the data into 10% test and 90% train      #
                                                                                                              #
    linear = linear_model.LinearRegression()                                                                  #
                                                                                                              #
    linear.fit(x_train, y_train)                                                                              #
    accuracy = linear.score(x_test, y_test)                                                                   #
    #print("Accuracy:",accuracy)                                                                               #
    TotalAccuracy += accuracy
    if accuracy > best: #this portion saves the best prediction after 30 run throughs                         #
        best = accuracy                                                                                       #
        with open("Data/studentmodel.pickle", "wb") as f: #this saves a file in our current directory and studentmodel.pickle (the wb means to write a new one if it doesn't already exist) and saves the prediction model in it #
            pickle.dump(linear, f)
        filename = 'Data/finalized_model.sav'
        pickle.dump(linear, open(filename, 'wb'))

PickledRegressionLine = pickle.load(open("Data/studentmodel.pickle", "rb")) #loads the prediction model into the variable "linear"
PickleModelAccuracy = PickledRegressionLine.score(x_test, y_test)
print("Current Pickle Model Accuracy:", PickleModelAccuracy)

TotalPickleModelAccuracy = 0
for _ in range(Runtimes):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,test_size=0.1)  # this randomly places all the data into 10% test and 90% train

    pickle_in = open("Data/studentmodel.pickle", "rb")  # loads the prediction model into the variable "linear"
    CurrentPickleModel = pickle.load(pickle_in)
    PickleModelAccuracy = CurrentPickleModel.score(x_test, y_test)
    TotalPickleModelAccuracy += PickleModelAccuracy
    #print("Current Pickle Model Accuracy:", PickleModelAccuracy)


print("\nCurrent Model Average Accuracy:", TotalAccuracy/Runtimes)
print("Current Pickle Model Average Accuracy:", TotalPickleModelAccuracy/Runtimes)
#ToDo automate the process for finding the best variables to use to make the model the most accurate
Sept13Accuracy = 0.830412223515