import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import metrics

digits = load_digits()

print('Image Data Shape', digits.data.shape)
print('Label Data Shape', digits.target.shape)

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize= 20)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

print(logisticRegr.predict(x_test[0:10]))

predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)

# How to predict a specific item or value
# logisticRegr.predict('insert_value')

# How to get the probability of a prediction being one or another value
print('Probabilities:\n', logisticRegr.predict_proba(x_test))

# Confusion matrix: this shows how many predictions were off, the diagonal is correct guesses, the sum of matrix values is equal to the total number of values in the test data set
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# Creates a heatmap for viewing the confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

# Shows specific images and their predicted vs actual scores
index = 0
classifiedIndex = []
for predict, actual in zip(predictions, y_test):
    if predict == actual:
        classifiedIndex.append(index)
    index += 1
plt.figure(figsize=(20,3))
for plotIndex, wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1,4, plotIndex + 1)
    plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual {}'.format(predictions[wrong], y_test[wrong]), fontsize=20)
plt.show()


