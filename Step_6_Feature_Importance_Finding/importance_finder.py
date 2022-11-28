import matplotlib.pyplot as plt
import rfpimp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from Extras.functions import time_formatter
import Extras.functions
import pickle

pd.options.display.width = 500
pd.set_option('display.max_columns', 80)
scaled_df = pd.read_csv('../Data/scaled_dataframe.csv')
print('Scaled Df:\n', scaled_df)


feature_length_wanted = 10

def importance_time_predictor(runtimes):
    predicted_time = runtimes * (0.8378246674 ** feature_length_wanted)
    return predicted_time

def feature_importer(runtimes, scaled_df, target_variable):
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
    rf = RandomForestRegressor(n_estimators=runtimes, n_jobs=-1)
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
        importance_dictionary[i] = imp['Importance'][loop_number]
        loop_number = loop_number + 1

    sorted_dict = dict(sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True))

    most_important_features = []
    for n in range(feature_length_wanted):
        new_corr_list = list(sorted_dict)
        # print('new corr list:', new_corr_list)
        most_important_features.append(new_corr_list[n])
    most_important_values = list(sorted_dict.values())
    print('Most Important Values:', most_important_values)
    most_important_features.insert(0, target_variable)

    log_appender(runtimes, most_important_features, target_variable)

    print('Running Plotter...')
    importance_plotter(most_important_features, most_important_values)

    return most_important_features

def feature_importer_non_printing(runtimes, scaled_df, target_variable):
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
    rf = RandomForestRegressor(n_estimators=runtimes, n_jobs=-1)
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
        importance_dictionary[i] = imp['Importance'][loop_number]
        loop_number = loop_number + 1

    sorted_dict = dict(sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True))

    most_important_features = []
    print('list(sorted dict):', list(sorted_dict))
    print('feature length wanted:', feature_length_wanted)
    for n in range(feature_length_wanted):
        new_corr_list = list(sorted_dict)
        print('new corr list[n]:', new_corr_list[n])
        most_important_features.append(new_corr_list[n]) # ToDo problem line
    most_important_values = list(sorted_dict.values())
    most_important_features.insert(0, target_variable)

    with open('../Data/most_recent_important_features.pickle', 'wb') as f:
        pickle.dump(most_important_features, f)

    log_appender(runtimes, most_important_features, target_variable)

    importance_plotter(most_important_features, most_important_values)


    return most_important_features


def importance_plotter(most_important_features, most_important_values):
    fig, ax = plt.subplots(figsize=(20, 18))
    ax.barh(most_important_features[1:], most_important_values[:feature_length_wanted], height=0.8, facecolor='grey', alpha=0.8, edgecolor='k')
    ax.set_xlabel('Importance score')
    ax.set_title('Permutation feature importance')
    ax.text(0.8, 0.15, 'aegis4048.github.io', fontsize=12, ha='center', va='center',
            transform=ax.transAxes, color='grey', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.get_current_fig_manager().window.state('zoomed')
    fig.tight_layout()
    plt.show()


def log_appender(runtimes, most_important_features, target_variable):
    with open('importance_finder_log.txt', 'a') as file:
        list_to_store = ['\n\nRuntimes:', str(runtimes), '\nTarget Variable:', target_variable, '\nMost Important Features:', str(most_important_features),
        '\nDate:', str(time.asctime())]
        stored_list_to_string = (' '.join(list_to_store))
        file.write(stored_list_to_string)



