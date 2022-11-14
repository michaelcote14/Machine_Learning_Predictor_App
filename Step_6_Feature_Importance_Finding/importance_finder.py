import matplotlib.pyplot as plt
import rfpimp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from Step_5_Scaling.scaler import scaled_df
from Step_1_Visualizing.visualization import target_variable
import pickle
import time
from Extras.functions import time_formatter, text_file_appender


print('Scaled Df:\n', scaled_df)
pickle_saver_activation = 'on'
runthroughs = 1000
feature_length_wanted = 23
predicted_time = runthroughs * (0.8378246674**feature_length_wanted)
print('Predicted Time:', time_formatter(predicted_time))

def feature_importer():
    ######################################## Data preparation #########################################

    pd.set_option('display.max_columns', 85)
    features = scaled_df.columns.tolist()

    ######################################## Train/test split #########################################

    scaled_df_train, scaled_df_test = train_test_split(scaled_df, test_size=0.20, random_state=0)
    scaled_df_train = scaled_df_train[features]
    scaled_df_test = scaled_df_test[features]

    X_train, y_train = scaled_df_train.drop(target_variable, axis=1), scaled_df_train[target_variable]
    X_test, y_test = scaled_df_test.drop(target_variable, axis=1), scaled_df_test[target_variable]

    # ################################################ Train #############################################
    #
    rf = RandomForestRegressor(n_estimators=runthroughs, n_jobs=-1)
    rf.fit(X_train, y_train)
    #
    # ############################### Permutation feature importance #####################################
    #
    imp = rfpimp.importances(rf, X_test, y_test)

    # turn this into a dictionary
    importance_list = imp.index.tolist()
    importance_dictionary = {}
    loop_number = 0
    for i in importance_list:
        print(i.rjust(20), ':',  str(imp['Importance'][loop_number]).ljust(100))
        importance_dictionary[i] = imp['Importance'][loop_number]
        loop_number = loop_number + 1

    sorted_dict = dict(sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True))
    most_important_features = []
    for n in range(feature_length_wanted):
        new_corr_list = list(sorted_dict)
        most_important_features.append(new_corr_list[n])
    most_important_features.insert(0, target_variable)
    if pickle_saver_activation.lower() == 'on':
        with open("most_important_features.pickle", "wb") as fp:
            pickle.dump(most_important_features, fp)
            print('-----------Pickle File Saved---------------')
        text_data_list = ['\n\nMost Important Features:', most_important_features, 'N_Estimators:', str(runthroughs), 'Date Completed:', str(time.asctime())]
        string_data_list = (', '.join(text_data_list))
        text_file_appender('importance_finder_data.txt', string_data_list)

    else:
        pass

    importance_plotter(imp)
    return most_important_features

def feature_importer_non_printing(feature_length_wanted=23):

    ######################################## Data preparation #########################################

    features = scaled_df.columns.tolist()

    ######################################## Train/test split #########################################

    scaled_df_train, scaled_df_test = train_test_split(scaled_df, test_size=0.20, random_state=0)
    scaled_df_train = scaled_df_train[features]
    scaled_df_test = scaled_df_test[features]

    X_train, y_train = scaled_df_train.drop(target_variable, axis=1), scaled_df_train[target_variable]
    X_test, y_test = scaled_df_test.drop(target_variable, axis=1), scaled_df_test[target_variable]

    # ################################################ Train #############################################
    #
    rf = RandomForestRegressor(n_estimators=runthroughs, n_jobs=-1)
    rf.fit(X_train, y_train)
    #
    # ############################### Permutation feature importance #####################################
    #
    imp = rfpimp.importances(rf, X_test, y_test)

    importance_list = imp.index.tolist()
    importance_dictionary = {}
    loop_number = 0
    for i in importance_list:
        importance_dictionary[i] = imp['Importance'][loop_number]
        loop_number = loop_number + 1

    sorted_dict = dict(sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True))
    most_important_features = []
    for n in range(feature_length_wanted):
        new_corr_list = list(sorted_dict)
        most_important_features.append(new_corr_list[n])
    most_important_features.insert(0, target_variable)
    if pickle_saver_activation.lower() == 'on':
        with open("most_important_features.pickle", "wb") as fp:   #rerun this to save the best pickle file, then comment this out
            pickle.dump(most_important_features, fp)
            print('-----------Pickle File Saved---------------')
        text_data_list = ['\n\nMost Important Features:', most_important_features, 'N_Estimators:',
                          str(runthroughs), 'Date Completed:', str(time.asctime())]
        string_data_list = (', '.join(text_data_list))
        text_file_appender('importance_finder_data.txt', string_data_list)
    return most_important_features


def importance_plotter(imp):
    # ToDo only plot the top X amount of most important features
    # ToDo use the saved pickle file's top features?
    ######################################## Data preparation #########################################

    features = feature_importer_non_printing(23)
    print('Features:\n', features)

    ############################################## Plot ################################################

    fig, ax = plt.subplots(figsize=(20, 18))

    ax.barh(imp.index, imp['Importance'], height=0.8, facecolor='grey', alpha=0.8, edgecolor='k')
    ax.set_xlabel('Importance score')
    ax.set_title('Permutation feature importance')
    ax.text(0.8, 0.15, 'aegis4048.github.io', fontsize=12, ha='center', va='center',
            transform=ax.transAxes, color='grey', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.get_current_fig_manager().window.state('zoomed')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    feature_importer()
    print('Actual Time:', time_formatter(time.time() - start_time))
    print('Predicted Time:', time_formatter(predicted_time))


# 22 most important features after 500,000 n_estimators:
# Target + Top Features: ['G3', 'G2', 'absences', 'studytime', 'G1', 'age', 'health', 'reason_home', 'activities_yes', 'traveltime', 'schoolsup_yes', 'paid_yes', 'nursery_yes', 'Fedu', 'Dalc', 'Mjob_teacher', 'Fjob_other', 'sex_M', 'Mjob_health', 'Medu', 'Pstatus_T', 'higher_yes', 'Fjob_teacher']
