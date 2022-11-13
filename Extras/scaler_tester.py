from Step_5_Scaling.scaler import scaled_df
import sklearn
import numpy as np
import time
from Step_3_Multiple_Encoding.multiple_hot_encoder import multiple_encoded_df
from Step_4_Data_Cleaning.data_cleaner import target_variable
from sklearn import linear_model


start_time = time.time()
RUNTIMES = 10000

def unscaled_score_finder():
    FEATURES = multiple_encoded_df.drop([target_variable], axis=1)
    X = np.array(FEATURES)
    y = np.array(multiple_encoded_df[target_variable])
    total_accuracy = 0
    for i in range(RUNTIMES):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(X_train, y_train)
        accuracy = linear.score(X_test, y_test)
        total_accuracy += accuracy
    unscaled_average_accuracy = total_accuracy / RUNTIMES
    return unscaled_average_accuracy

def scaled_score_finder():
    SCALED_FEATURES = scaled_df.drop([target_variable], axis=1)
    SCALED_X = np.array(SCALED_FEATURES)
    SCALED_Y = np.array(scaled_df[target_variable])
    total_accuracy = 0
    for i in range(RUNTIMES):
        SCALED_X_train, SCALED_X_test, SCALED_Y_train, SCALED_Y_test = sklearn.model_selection.train_test_split(
            SCALED_X, SCALED_Y, test_size=0.2)
        linear = linear_model.LinearRegression()

        linear.fit(SCALED_X_train, SCALED_Y_train)
        accuracy = linear.score(SCALED_X_test, SCALED_Y_test)
        total_accuracy += accuracy
    scaled_average_accuracy = total_accuracy / RUNTIMES
    return scaled_average_accuracy


if __name__ == "__main__":
    scaled_average_accuracy = scaled_score_finder()
    print('\nScaled Average Accuracy:', scaled_average_accuracy)
    unscaled_average_accuracy = unscaled_score_finder()
    print('Unscaled Average Accuracy:', unscaled_average_accuracy)
    print('Scaled Average Accuracy - Unscaled Average Accuracy =', scaled_average_accuracy - unscaled_average_accuracy )
