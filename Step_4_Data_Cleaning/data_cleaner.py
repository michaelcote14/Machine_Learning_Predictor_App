from Step_3_Multiple_Encoding.multiple_hot_encoder import multiple_encoded_df




def outlier_cleaner():
    for feature in multiple_encoded_df:
        Q1 = multiple_encoded_df[feature].quantile(0.25)
        Q3 = multiple_encoded_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        print('Q1:', Q1, '\nQ3:', Q3, '\nIQR:', IQR)

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        print('upper limit:', upper_limit)

        outlier_free_df = multiple_encoded_df.loc[(multiple_encoded_df[feature] < upper_limit) & (multiple_encoded_df[feature] > lower_limit)]

        print('Before removing outliers:', len(multiple_encoded_df))
        print('After removing outliers:', len(outlier_free_df))
        print('Total outliers:', len(multiple_encoded_df) - len(outlier_free_df))
        percent_removed = (len(multiple_encoded_df) - len(outlier_free_df)) / len(multiple_encoded_df)
        print('Percent Removed:', percent_removed)
    if percent_removed < 0.05:
        print("---Changing this column's data by capping---")

        # capping
        outlier_free_df = multiple_encoded_df.copy()
        outlier_free_df.loc[(outlier_free_df[feature] > upper_limit), feature] = upper_limit
        outlier_free_df.loc[(outlier_free_df[feature] < lower_limit), feature] = lower_limit
        print('multiple_encoded_df column mean:', multiple_encoded_df[feature].mean())
        print('outlier_free_df column mean:', outlier_free_df[feature].mean())
    else:
        pass
    print(outlier_free_df)

    return outlier_free_df


def outlier_cleaner_non_printing():
    for feature in multiple_encoded_df:
        print('Feature:', feature)
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
    print('Null Data:', data_to_null_clean.isnull().sum())
    fully_cleaned_df = data_to_null_clean.fillna(data_to_null_clean.mean(numeric_only=True))
    return fully_cleaned_df

def null_column_cleaner_non_printing(data_to_null_clean):
    fully_cleaned_df = data_to_null_clean.fillna(data_to_null_clean.mean(numeric_only=True))
    return fully_cleaned_df

def null_row_cleaner(data_to_clean):
    fully_cleaned_df = data_to_clean.dropna(how='all')
    return fully_cleaned_df


def unnamed_column_dropper(data_to_unname_drop):
    for column in data_to_unname_drop.columns:
        if column.startswith('Unnamed'):
            data_to_unname_drop.drop([column], axis=1, inplace=True)
    return data_to_unname_drop


def full_cleaner():
    outlier_free_df = outlier_cleaner_non_printing()
    null_column_free_df = null_column_cleaner_non_printing(outlier_free_df)
    null_free_df = null_row_cleaner(null_column_free_df)
    fully_cleaned_df = unnamed_column_dropper(null_free_df)
    return fully_cleaned_df


fully_cleaned_df = full_cleaner()
print('Fully Cleaned Df:\n', fully_cleaned_df)
if __name__ == '__main__':
    pass

