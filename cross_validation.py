from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import one_hot_encoder_multiple_categories
import numpy as np
import sklearn

dataframe = one_hot_encoder_multiple_categories.encoded_df
data = dataframe[['G3', 'G2', 'G1', 'age', 'goout', 'romantic_yes', 'traveltime', 'paid_yes', 'internet_yes', 'studytime']]
target_variable = 'G3'

X = np.array(data.drop([target_variable], axis=1))
y = np.array(data[target_variable])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
my_linear_regression = linear_model.LinearRegression().fit(X_train, y_train)

y_pred = my_linear_regression.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
for i in range(20):
    print('Cross Val Score:', cross_val_score(my_linear_regression, X, y, cv=i+2).mean())
