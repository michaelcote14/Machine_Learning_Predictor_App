def correlator():
    from scaler import main
    dataframe = main()
    print('dataframe:\n', dataframe)

    correlation_dataframe = dataframe.corr()
    print('correlation dataframe:', correlation_dataframe)
    target_feature = 'G3'
    target_index = correlation_dataframe.columns.get_loc(target_feature)
    correlation_dictionary = {}
    row_runner = 0
    for column in dataframe:
        left_column = correlation_dataframe.columns[row_runner]
        print('\nLeft Column:', left_column)
        correlation_to_target = correlation_dataframe.iloc[target_index, row_runner]
        print('Correlation to Target:', correlation_dataframe.iloc[target_index, row_runner])
        # put dictionary here
        correlation_dictionary[left_column] = abs(correlation_to_target)

        row_runner = row_runner + 1
    print('\nDictionary of Correlations:\n', correlation_dictionary)

    sorted_dict = dict(sorted(correlation_dictionary.items(), key=lambda x: x[1], reverse=True)) #problem line
    top_correlators = []
    for n in range(23):
        new_corr_list = list(sorted_dict)
        top_correlators.append(new_corr_list[n])
    print('Top 23 Correlators', top_correlators)
    return top_correlators


if __name__ == '__main__':
    correlator()
