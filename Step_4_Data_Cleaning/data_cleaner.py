def outlier_cleaner(multiple_encoded_df):
    for feature in multiple_encoded_df:
        Q1 = multiple_encoded_df[feature].quantile(0.25)
        Q3 = multiple_encoded_df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outlier_free_df = multiple_encoded_df.loc[(multiple_encoded_df[feature] < upper_limit) & (multiple_encoded_df[feature] > lower_limit)]

        percent_removed = (len(multiple_encoded_df) - len(outlier_free_df)) / len(multiple_encoded_df)
        if percent_removed < 0.05:
            # capping
            outlier_free_df = multiple_encoded_df.copy()
            outlier_free_df.loc[(outlier_free_df[feature] > upper_limit), feature] = upper_limit
            outlier_free_df.loc[(outlier_free_df[feature] < lower_limit), feature] = lower_limit
        else:
            pass

        return outlier_free_df


def null_column_cleaner(data_to_null_clean):
    for column in data_to_null_clean.columns:
        if (data_to_null_clean[column].isnull().sum() / (len(data_to_null_clean))) > 0.25:
            print('column with too many null values deleted:', column)
            data_to_null_clean.drop(column, inplace=True, axis=1)

    fully_cleaned_df = data_to_null_clean.fillna(data_to_null_clean.mean(numeric_only=True))
    return fully_cleaned_df


def null_row_cleaner(data_to_clean):
    fully_cleaned_df = data_to_clean.dropna(how='all')
    fully_cleaned_df.reset_index(drop=True, inplace=True)
    return fully_cleaned_df


def unnamed_column_dropper(data_to_unname_drop):
    for column in data_to_unname_drop.columns:
        if column.startswith('Unnamed'):
            data_to_unname_drop.drop([column], axis=1, inplace=True)
    return data_to_unname_drop


def full_cleaner(multiple_encoded_df):
    outlier_free_df = outlier_cleaner(multiple_encoded_df)
    null_column_free_df = null_column_cleaner(outlier_free_df)
    # null_free_df = null_row_cleaner(null_column_free_df)
    fully_cleaned_df = unnamed_column_dropper(null_column_free_df)
    return fully_cleaned_df


if __name__ == '__main__':
    pass

