import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
original_df = pd.read_csv("../Data/nfl_data.csv")
target_variable = 'pass_blitzed'


pd.options.display.width = 500
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 80)
print(original_df)


def data_type_cleaner():
    for column in original_df.columns:
        if original_df.dtypes[column] == 'bool':
            original_df[column] = original_df[column].astype(str)
    type_clean_df = original_df
    return type_clean_df


def box_and_whisker(dataframe):
    # make a box and whisker
    num_of_groupings = math.ceil(len(original_df.columns) / 15)
    print('Length/15:', num_of_groupings)
    start, finish = 0, 15
    for _ in range(num_of_groupings):
        shrunken_df = original_df.iloc[:, start:finish]
        if target_variable not in shrunken_df.columns.tolist():
            shrunken_df = pd.concat([original_df[target_variable], shrunken_df], axis=1)
        bw = sns.boxplot(shrunken_df[shrunken_df.columns.tolist()])
        bw = sns.stripplot(shrunken_df[shrunken_df.columns.tolist()], palette='dark:red', size=2, jitter=0.25)
        plt.xticks(rotation=25)
        for lab in bw.get_xticklabels():
            if lab.get_text() == target_variable:
                lab.set_fontweight('bold')
                lab.set_color('blue')
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
        start += 15
        finish += 15


def histogram(dataframe):
    # make a histogram
    mean = dataframe.mean(numeric_only=True)[target_variable]
    x = dataframe[target_variable].values
    sns.histplot(x, kde=True, color='blue')
    plt.axvline(mean, 0, 1, color='red')
    plt.ylabel('# of category')
    plt.xlabel(target_variable)
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()

def heatmap(dataframe):
    # How to make a heatmap
    num_of_groupings = math.ceil(len(original_df.columns) / 25)
    print('Length/30:', num_of_groupings)
    start, finish = 0, 25
    for _ in range(num_of_groupings):
        shrunken_df = original_df.iloc[:, start:finish]
        if target_variable not in shrunken_df.columns.tolist():
            shrunken_df = pd.concat([shrunken_df, original_df[target_variable]], axis=1)
        correlation = shrunken_df.corr()
        matrix = np.triu(correlation)
        colormap = sns.color_palette('Reds')
        hm = sns.heatmap(correlation, annot = True, cmap = colormap, mask=matrix) # YlGnBu is best color
        # hm.set_yticklabels()[0].set_color('red')
        for lab in hm.get_yticklabels():
            if lab.get_text() == target_variable:
                lab.set_fontweight('bold')
                lab.set_color('blue')
        plt.xlabel(target_variable, fontweight='bold')
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
        start += 25
        finish += 25


type_clean_df = data_type_cleaner()
if __name__ == '__main__':
    # box_and_whisker(original_df)
    # histogram(original_df)
    # heatmap(original_df)
    pass




