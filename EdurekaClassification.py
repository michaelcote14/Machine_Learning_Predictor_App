import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("student-mat.csv", sep=";")
print(data.head()) #prints first few columns

print(data.shape) #shows full shape of data
print(data['age'].unique()) #shows all the unique numbers in the 'age' column
print(data.groupby('age').size()) #shows the numbers for each unique age. Kind of like a histogram.

#this sections shows a histogram for all of the G3 scores
import seaborn as sns
sns.countplot(data['G3'],label="Count")
plt.show()

# #this section is supposed to make a box plot, but it doesn't work for some reason
# data.drop('G3',axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9),title='G1')
# plt.savefig('Test_Box')
# plt.show()

#this section is supposed to plot histograms for each variable, but it also doesn't work for some reason
import pylab as pl
data.drop('G3', axis=1).hist(bins=30, figsize= (9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('Scores_hist')
plt.show


#this section divides the data into target and predictor variables
feature_names = ['age', 'G1', 'G2', 'absences'] #predictor variables
X = data[feature_names]
y = data['G3'] #target variable



#this section splits the data into test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Logistic regression algorithm for classification and shows how accurate the model is
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(logreg.score(X_test, y_test)))

#Decision tree algorithm for classification
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree Classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree Classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

#KNN Classifier algorithm for classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN Classifier on training set: {:.2f}'
      .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN Classifier on test set: {:.2f}'
      .format(knn.score(X_test, y_test)))

#NaiveBayes Classifer algorithm for classification
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set:')




