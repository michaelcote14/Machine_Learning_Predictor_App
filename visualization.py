import numpy as np #this is for doing interesting things wtih numbers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from importance_finder import importance_plotter


original_df = pd.read_csv("Data/student-mat.csv")
target_variable = 'G3'

# make a box and whisker
sns.boxplot(original_df[original_df.columns.tolist()])
sns.stripplot(original_df[original_df.columns.tolist()], color='red', size=2, jitter=0.25)
plt.get_current_fig_manager().window.state('zoomed')
plt.show()


# make a histogram
mean = original_df.mean(numeric_only=True)[target_variable]
x = original_df[target_variable].values
sns.histplot(x, kde=True, color='blue')
plt.axvline(mean, 0, 1, color='red')
plt.ylabel('# of category')
plt.xlabel(target_variable)
plt.get_current_fig_manager().window.state('zoomed')
plt.show()

# How to make a heatmap
correlation = original_df.corr()
hm = sns.heatmap(correlation, annot = True, cmap = 'YlGnBu') # YlGnBu is best color
# how to make all plots immediately go to full screen
plt.get_current_fig_manager().window.state('zoomed')
plt.show()




for column in original_df:
    unique_vals = np.unique(original_df[column])
    nr_values = len(unique_vals)
    if nr_values < 10:
        print('Values for', column.rjust(10), ':', nr_values, '--', unique_vals)
    else:
        print('Values for', column.rjust(10), ':', nr_values)

importance_plotter()



