def feature_importer():
    import rfpimp
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import multiple_hot_encoder
    from sklearn.inspection import permutation_importance

    ######################################## Data preparation #########################################

    df = multiple_hot_encoder.multiple_encoder()
    target_variable = 'G3'
    features = df.columns.tolist()
    print('Features:\n', features)

    ######################################## Train/test split #########################################

    df_train, df_test = train_test_split(df, test_size=0.20, random_state=0)
    df_train = df_train[features]
    print('df train:\n', df_train)
    print('df:\n', df)
    df_test = df_test[features]
    print('df test:\n', df_test)

    X_train, y_train = df_train.drop(target_variable, axis=1), df_train[target_variable] # problem line
    X_test, y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

    # ################################################ Train #############################################
    #
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    rf.fit(X_train, y_train)
    #
    # ############################### Permutation feature importance #####################################
    #
    imp = rfpimp.importances(rf, X_test, y_test)
    print(imp)

    # turn this into a dictionary
    importance_list = imp.index.tolist()
    print(importance_list)
    print(imp['Importance'][0])
    importance_dictionary = {}
    loop_number = 0
    for i in importance_list:
        print('\n', i, imp['Importance'][loop_number])
        importance_dictionary[i] = imp['Importance'][loop_number]
        loop_number = loop_number + 1

    print(importance_dictionary)
    sorted_dict = dict(sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True)) #problem line
    most_important_features = []
    for n in range(23):
        new_corr_list = list(sorted_dict)
        most_important_features.append(new_corr_list[n])
    print('Top 23 Important Features', most_important_features)
    most_important_features.append(target_variable)
    print('new important features:', most_important_features)
    return most_important_features


def importance_plotter():
    import rfpimp
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import multiple_hot_encoder
    from sklearn.inspection import permutation_importance

    ######################################## Data preparation #########################################

    df = multiple_hot_encoder.multiple_encoder()
    target_variable = 'G3'
    features = df.columns.tolist()
    print('Features:\n', features)

    ######################################## Train/test split #########################################

    df_train, df_test = train_test_split(df, test_size=0.20, random_state=0)
    df_train = df_train[features]
    print('df train:\n', df_train)
    print('df:\n', df)
    df_test = df_test[features]
    print('df test:\n', df_test)

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

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    feature_importer()