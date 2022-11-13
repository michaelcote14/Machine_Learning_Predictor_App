import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle
import time
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
from Step_1_Visualizing.visualization import target_variable



predictor_data_dict = {'age': [16.0], 'G2': [10.0], 'goout': [3], 'internet_yes': [1]}

start_time = time.time()
run_evaluation = 'yes'
scaled_df = pd.read_csv('../Step_5_Scaling/scaled_dataframe.csv')
with open('../Data/most_important_features.pickle', 'rb') as f:
    most_important_features = pickle.load(f)[:]
df = scaled_df[most_important_features]
print('scaled_df\n', scaled_df)

X = np.array(df.drop([target_variable], axis=1), dtype='object')
y = np.array(df[target_variable], dtype='object')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)


def predictor_scaler():
    from Step_5_Scaling.scaler import predictor_data_scaler
    scaled_predictor_array, scaled_predictor_df = predictor_data_scaler(predictor_data_dict)
    mean_dataframe = pd.DataFrame(df.mean())
    mean_dataframe = mean_dataframe.T
    # plug in our data to the above dataframe
    mean_dataframe.update(scaled_predictor_df)
    mean_dataframe.drop([target_variable], axis=1, inplace=True)
    # now turn the above into an array
    finalized_predictor_array = np.array(mean_dataframe)
    return finalized_predictor_array


def predictor():
    finalized_predictor_array = predictor_scaler()

    all_current_model_predictions = MyLinearRegression.predict(X_test)
    current_model_input_prediction = MyLinearRegression.predict(finalized_predictor_array)
    current_cross_val_score = cross_val_score(MyLinearRegression, X, y, cv=10).mean()
    current_normal_score = MyLinearRegression.score(X_test, y_test)


    pickle_in = open('../Data/studentmodel.pickle', 'rb')
    old_pickled_regression_line = pickle.load(pickle_in)
    all_pickle_model_predictions = old_pickled_regression_line.predict(X_test)
    pickle_model_input_prediction = old_pickled_regression_line.predict(finalized_predictor_array)
    pickle_cross_val_score = cross_val_score(old_pickled_regression_line, X, y, cv=10).mean()
    pickle_normal_score = old_pickled_regression_line.score(X_test, y_test)
    pickle_mean_absolute_error = metrics.mean_absolute_error(y_test, all_pickle_model_predictions)
    pickle_r2_score = r2_score(y_test, all_pickle_model_predictions)


    print(':             Statistic                :              Current Model                :        Pickle Model       ')
    nested_list = [[('Target Prediction: ' + target_variable), current_model_input_prediction, pickle_model_input_prediction],
                   ['Accuracy', current_cross_val_score, pickle_cross_val_score],
                   ['Mean Absolute Error',  metrics.mean_absolute_error(y_test, all_current_model_predictions), pickle_mean_absolute_error],
                   ['R2 Score', r2_score(y_test, all_current_model_predictions),
                    pickle_r2_score]]
    for item in nested_list:
        print(':', item[0], ' '*(35-len(item[0])), ':', item[1],  ' '*(40-len(str(item[1]))),
              ':', item[2],  ' '*(20-len(str(item[2]))))
    print(': Range                                :', current_model_input_prediction
          - current_cross_val_score * 0.01 * current_model_input_prediction,
          '-',
          current_cross_val_score * 0.01 * current_model_input_prediction
          + current_model_input_prediction, '              :', pickle_model_input_prediction
          - pickle_cross_val_score * 0.01 * pickle_model_input_prediction,
          '-',
          pickle_model_input_prediction
          + pickle_cross_val_score * 0.01 * pickle_model_input_prediction)

    print(': Cross-Val Difference                 :', format(current_cross_val_score - current_normal_score, '.17f'), '                     :',
      pickle_cross_val_score - pickle_normal_score)
    print('Positive number above means cross val score was higher, which means your model is overfitting')

    Sum, Max = 0, 0
    if run_evaluation.lower() == 'yes':
        print('    Predicted             Actual', 'Difference'.rjust(21))
        for x in range(len(all_current_model_predictions)):
            print(str(all_current_model_predictions[x]).ljust(23), str(y_test[x]).rjust(5),
                  str(y_test[x] - all_current_model_predictions[x]).rjust(30))
            IndividualDifference = abs(y_test[x] - all_current_model_predictions[x])
            Sum = Sum + IndividualDifference
            if IndividualDifference > Max:
                Max = IndividualDifference



    else:
        pass
    return all_current_model_predictions, all_pickle_model_predictions

def predictor_plotter(all_current_model_predictions):
    plt.figure(figsize=(15, 10))
    plt.scatter(y_test, all_current_model_predictions, c='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('Actual Values for ' + target_variable)
    plt.ylabel('Predicted Values for ' + target_variable)
    plt.title('Actual Vs Predicted Values')
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()



if __name__ == '__main__':
    from Step_5_Scaling.scaler import scaled_predictor_array, scaled_predictor_df
    all_current_model_predictions, all_pickle_model_predictions = predictor()
    predictor_plotter(all_current_model_predictions)


