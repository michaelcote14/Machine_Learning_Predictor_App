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

def crosser(regression_line_to_scale, splits_to_make=20):
    my_cross_val_score_best = 0
    for cross_count in range(splits_to_make):
        my_cross_val_score = cross_val_score(regression_line_to_scale, X, y, cv=cross_count+2).mean()
        if my_cross_val_score > my_cross_val_score_best:
            best_cross_count = cross_count
            my_cross_val_score_best = my_cross_val_score
    return best_cross_count, my_cross_val_score_best
