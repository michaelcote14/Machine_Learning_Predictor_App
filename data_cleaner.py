



def outlier_cleaner():
    from multiple_hot_encoder import multiple_encoder

    encoded_df = multiple_encoder()
    print('encoded df:\n', encoded_df.head)
    for feature in encoded_df:
        print('\nfeature:', feature)
        Q1 = encoded_df[feature].quantile(0.25)
        Q3 = encoded_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        print('Q1:', Q1, '\nQ3:', Q3, '\nIQR:', IQR)

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        print('upper limit:', upper_limit)

        outlier_free_df = encoded_df.loc[(encoded_df[feature] < upper_limit) & (encoded_df[feature] > lower_limit)]

        print('Before removing outliers:', len(encoded_df))
        print('After removing outliers:', len(outlier_free_df))
        print('Total outliers:', len(encoded_df) - len(outlier_free_df))
        percent_removed = (len(encoded_df) - len(outlier_free_df)) / len(encoded_df)
        print('Percent Removed:', percent_removed)
        if percent_removed < 0.05:
            print("---Changing this column's data by capping---")

            # capping
            outlier_free_df = encoded_df.copy()
            outlier_free_df.loc[(outlier_free_df[feature] > upper_limit), feature] = upper_limit
            outlier_free_df.loc[(outlier_free_df[feature] < lower_limit), feature] = lower_limit
            print('encoded_df column mean:', encoded_df[feature].mean())
            print('outlier_free_df column mean:', outlier_free_df[feature].mean())
        else:
            pass
        print(outlier_free_df)

        return outlier_free_df


def null_value_cleaner(data_to_null_clean):
    print('Null Data:', data_to_null_clean.isnull().sum())
    fully_cleaned_df = data_to_null_clean.fillna(data_to_null_clean.mean())
    return fully_cleaned_df


def unnamed_column_dropper(data_to_unname_drop):
    for column in data_to_unname_drop.columns:
        if column.startswith('Unnamed'):
            data_to_unname_drop.drop([column], axis=1, inplace=True)
    return data_to_unname_drop


def full_cleaner():
    outlier_free_df = outlier_cleaner()
    null_free_df = null_value_cleaner(outlier_free_df)
    fully_cleaned_df = unnamed_column_dropper(null_free_df)
    return fully_cleaned_df



if __name__ == '__main__':
    fully_cleaned_df = full_cleaner()
