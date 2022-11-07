def feature_importer(feature_length_wanted=23):
    import rfpimp
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import pandas as pd
    ######################################## Data preparation #########################################

    df = pd.read_csv('scaled_dataframe.csv')
    target_variable = 'G3'
    features = df.columns.tolist()

    ######################################## Train/test split #########################################

    df_train, df_test = train_test_split(df, test_size=0.20, random_state=0)
    df_train = df_train[features]
    df_test = df_test[features]

    X_train, y_train = df_train.drop(target_variable, axis=1), df_train[target_variable]
    X_test, y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

    # ################################################ Train #############################################
    #
    rf = RandomForestRegressor(n_estimators=100000, n_jobs=-1)
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

    sorted_dict = dict(sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True)) #problem line
    most_important_features = []
    for n in range(feature_length_wanted):
        new_corr_list = list(sorted_dict)
        most_important_features.append(new_corr_list[n])
    most_important_features.insert(0, target_variable)
    print('\nTarget + Top Features:', most_important_features)
    return most_important_features

def feature_importer_non_printing(feature_length_wanted=23):
    import rfpimp
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import pandas as pd
    ######################################## Data preparation #########################################

    df = pd.read_csv('scaled_dataframe.csv')
    target_variable = 'G3'
    features = df.columns.tolist()

    ######################################## Train/test split #########################################

    df_train, df_test = train_test_split(df, test_size=0.20, random_state=0)
    df_train = df_train[features]
    df_test = df_test[features]

    X_train, y_train = df_train.drop(target_variable, axis=1), df_train[target_variable]
    X_test, y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

    # ################################################ Train #############################################
    #
    rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
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
    return most_important_features


def importance_plotter():
    import rfpimp
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from multiple_hot_encoder import multiple_encoder

    ######################################## Data preparation #########################################

    df = multiple_encoder()
    target_variable = 'G3'
    features = df.columns.tolist()

    ######################################## Train/test split #########################################

    df_train, df_test = train_test_split(df, test_size=0.20, random_state=0)
    df_train = df_train[features]
    df_test = df_test[features]

    X_train, y_train = df_train.drop(target_variable, axis=1), df_train[target_variable]  # problem line
    X_test, y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

    # ################################################ Train #############################################
    #
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    rf.fit(X_train, y_train)
    #
    # ############################### Permutation feature importance #####################################
    #
    imp = rfpimp.importances(rf, X_test, y_test)
    # negative importances can be considered as 0
    print(imp)
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
    feature_importer(22)
    # importance_plotter()
