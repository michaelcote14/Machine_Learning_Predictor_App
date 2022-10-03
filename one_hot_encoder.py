import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

pd.options.display.width = 500
pd.set_option('display.max_columns', 500)
dataframe = pd.read_csv("dataframe/student-mat.csv")




chosen_dataframe = dataframe[['sex']]
print('chosen_dataframe:\n', chosen_dataframe)


transformer = make_column_transformer((OneHotEncoder(), ['sex']), remainder='passthrough')
print('transformer:\n', transformer)
transformed = transformer.fit_transform(chosen_dataframe)
print('transformed:\n', transformed)
transformed_df = pd.dataframeFrame(transformed, columns=transformer.get_feature_names_out())
print('transformed_df:\n', transformed_df)
transformed_df.rename(columns={'onehotencoder__sex_M': 'sex_M'}, inplace=True) #ToDo automate this using reusable variables
print('transformed_df_renamed:\n', transformed_df)
print('transformed:\n', transformed_df.head())


column_to_drop = transformed_df.columns[0]
print('column_to_drop\n', column_to_drop)
# print('column_to_drop:', transformed_df.columns[0])
only_one_transformed = transformed_df.drop(columns=str(column_to_drop))
print('only one transformed:\n',only_one_transformed)

# merged = dataframe.merge(chosen_dataframe)
# print('non merged:\n', dataframe)
# print('merged:\n', merged.head)
#
# print('transformed_df:', transformed_df)
# print(dataframe)

#ToDo change the one hot encoder prefix to be a smaller name by taking out the onehotencoder in the beginning
#ToDo drop the sex column now that you have merged it
#ToDo make it easier to use by implementing more reusable variables
combined_frame = pd.concat([only_one_transformed, dataframe], axis = 1)
print('combined frame:\n', combined_frame)


#ToDo make it to where it can take more than one variable




