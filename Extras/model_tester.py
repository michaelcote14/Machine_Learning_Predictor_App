import pandas as pd
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge, OrthogonalMatchingPursuit
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
# import catboost
from sklearn.model_selection import KFold, cross_val_score


dataframe = pd.read_csv("C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/Saved_CSVs/nfl_data(numeric_only).csv")
target_variable = 'Wind_Speed'

X = np.array(dataframe.drop([target_variable], axis=1))
y = np.array(dataframe[target_variable])

regression_models = {
    'regression': sklearn.linear_model.LinearRegression(),
    'br': BayesianRidge(),
    'huber': HuberRegressor(),
    'ridge': Ridge(),
    'omp': OrthogonalMatchingPursuit(),
    'lightgbm': LGBMRegressor()
}

# Get the scores for each model
results_dict = {}
kf = KFold(n_splits=10)
for name, model in regression_models.items():
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
    results_dict[name] = score

sorted_results_dict = dict(sorted(results_dict.items(), key=lambda x:x[1], reverse=True))
for name, result in sorted_results_dict.items():
    print(name.ljust(10), ':', result)


# How to combine predictions
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

final_predictions = (
    0.16666667 * regression_models['regression'].predict(X_test) +
    0.16666667 * regression_models['br'].predict(X_test) +
    0.16666667 * regression_models['huber'].predict(X_test) +
    0.16666667 * regression_models['ridge'].predict(X_test) +
    0.16666667 * regression_models['omp'].predict(X_test) +
    0.16666667 * regression_models['lightgbm'].predict(X_test)
                    )

final_score = (
        0.16666667 * regression_models['regression'].score(X_test, y_test) +
        0.16666667 * regression_models['br'].score(X_test, y_test) +
        0.16666667 * regression_models['huber'].score(X_test, y_test) +
        0.16666667 * regression_models['ridge'].score(X_test, y_test) +
        0.16666667 * regression_models['omp'].score(X_test, y_test) +
        0.16666667 * regression_models['lightgbm'].score(X_test, y_test)
)

print('final score:', final_score)


x0 = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
from scipy.optimize import minimize
import numpy as np

def objective(x, sign=-1.0): # the sign makes this a maximized optimization instead
    final_score = (
            x[0] * regression_models['regression'].score(X_test, y_test) +
            x[1] * regression_models['br'].score(X_test, y_test) +
            x[2] * regression_models['huber'].score(X_test, y_test) +
            x[3] * regression_models['ridge'].score(X_test, y_test) +
            x[4] * regression_models['omp'].score(X_test, y_test) +
            x[5] * regression_models['lightgbm'].score(X_test, y_test)
    )

    return sign * final_score

print(objective(x0))

b = (0.0, 1.0)
bnds = (b, b, b, b, b, b)
con1 = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - 1})
cons = [con1]
solution = minimize(objective, x0, bounds=bnds, constraints=cons, method='SLSQP')

best_weights = solution.x
print('best weights:\n', best_weights)

print('best score:', -(solution.fun))

print('sum of weights:', np.sum(solution.x))

# Now plug in the weights
optimized_score = (
        best_weights[0] * regression_models['regression'].score(X_test, y_test) +
        best_weights[1] * regression_models['br'].score(X_test, y_test) +
        best_weights[2] * regression_models['huber'].score(X_test, y_test) +
        best_weights[3] * regression_models['ridge'].score(X_test, y_test) +
        best_weights[4] * regression_models['omp'].score(X_test, y_test) +
        best_weights[5] * regression_models['lightgbm'].score(X_test, y_test)
)

print('optimized score:', optimized_score)

weights_and_models_dict = {regression_models['regression']: best_weights[0],
                           regression_models['br']: best_weights[1],
                           regression_models['huber']: best_weights[2],
                           regression_models['ridge']: best_weights[3],
                           regression_models['omp']: best_weights[4],
                           regression_models['lightgbm']: best_weights[5]}
# Testing multiple pickle dump and retrievals
import pickle

with open('multiple_models.pickle', 'wb') as f:
    for model in regression_models.values():
        pickle.dump(weights_and_models_dict, f)

pickle_object = open('multiple_models.pickle', 'rb')
pickled_weights_and_models_dict = pickle.load(pickle_object)
print('pickled weights and models dict:\n', pickled_weights_and_models_dict)



#
# # Retrieval
# models = []
# with open('multiple_models.pickle', 'rb') as f:
#     while True:
#         try:
#             models.append(pickle.load(f))
#         except EOFError:
#             break
#
# # Using the pickle model
# print('models:', models)
# new_score = (
#     best_weights[0] * models[0].score(X_test, y_test) +
#     best_weights[1] * models[1].score(X_test, y_test) +
#     best_weights[2] * models[2].score(X_test, y_test) +
#     best_weights[3] * models[3].score(X_test, y_test) +
#     best_weights[4] * models[4].score(X_test, y_test) +
#     best_weights[5] * models[5].score(X_test, y_test)
#             )
#
# print('new score:', new_score)

