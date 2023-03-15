import matplotlib.pyplot as plt
import rfpimp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from Extras.functions import time_formatter
import Extras.functions
import pickle

class ImportanceFeaturesFinder:
    def importance_time_predictor(runtimes, amount_of_features):
        print('amount of features:', amount_of_features)
        predicted_time = int(runtimes) * (0.8378246674 ** int(amount_of_features))
        predicted_time = time_formatter(predicted_time)
        return predicted_time

    def feature_importer(runtimes, scaled_df, target_variable, feature_length_wanted):
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
        rf = RandomForestRegressor(n_estimators=int(runtimes), n_jobs=-1)
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
        for n in range(int(feature_length_wanted)):
            new_corr_list = list(sorted_dict)
            print('new corr list:', new_corr_list)
            most_important_features.append(new_corr_list[n])
        most_important_values = list(sorted_dict.values())
        print('Most Important Values:', most_important_values)
        most_important_features.insert(0, target_variable)

        log_appender(runtimes, most_important_features, target_variable)

        print('Running Plotter...')
        importance_plotter(most_important_features, most_important_values, feature_length_wanted)

        return most_important_features


    def importance_plotter(self, most_important_features, most_important_values, feature_length_wanted):
        fig, ax = plt.subplots(figsize=(20, 18))
        ax.barh(most_important_features[1:], most_important_values[:int(feature_length_wanted)], height=0.8, facecolor='grey', alpha=0.8, edgecolor='k')
        ax.set_xlabel('Importance score')
        ax.set_title('Permutation feature importance')
        ax.text(0.8, 0.15, 'aegis4048.github.io', fontsize=12, ha='center', va='center',
                transform=ax.transAxes, color='grey', alpha=0.5)
        plt.gca().invert_yaxis()
        plt.get_current_fig_manager().window.state('zoomed')
        fig.tight_layout()
        plt.show()


Best Average Accuracy: 0.42163469465356973
Best Features: ['Temperature', 'Wind_Speed', 'Temperature', 'Vegas_Line', 'Vegas_Line', 'Over_Under', 'Temperature']