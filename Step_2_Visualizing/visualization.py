import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.widgets import Cursor
from tkinter import *
from tkinter import ttk
import numpy as np
import math


class Grapher:
    def box_and_whisker(self, type_clean_df, target_variable):
        if len(type_clean_df.columns) > 15:
            graph_width = len(type_clean_df.columns)/1.5
        else:
            graph_width = 9

        figure, ax = plt.subplots(figsize=(graph_width, 10))
        bw = sns.boxplot(type_clean_df[type_clean_df.columns.tolist()])
        bw = sns.stripplot(type_clean_df[type_clean_df.columns.tolist()], palette='dark:red', size=2, jitter=0.25)
        plt.xticks(rotation=25)
        ax.grid(axis='y')
        for xtick in bw.get_xticklabels():
            if xtick.get_text() == target_variable:
                xtick.set_fontweight('bold')
                xtick.set_color('blue')


        return figure

    def histogram(self, type_clean_df, target_variable):
        # This makes values show on each bar
        def show_values(axs, orient="v", space=.01):
            def _single(ax):
                if orient == "v":
                    for p in ax.patches:
                        _x = p.get_x() + p.get_width() / 2
                        _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                        value = '{:.1f}'.format(p.get_height())
                        ax.text(_x, _y, value, ha="center")
                elif orient == "h":
                    for p in ax.patches:
                        _x = p.get_x() + p.get_width() + float(space)
                        _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                        value = '{:.1f}'.format(p.get_width())
                        ax.text(_x, _y, value, ha="left")

            if isinstance(axs, np.ndarray):
                for idx, ax in np.ndenumerate(axs):
                    _single(ax)
            else:
                _single(axs)

        # make a histogram
        mean = type_clean_df.mean(numeric_only=True)[target_variable]
        x = type_clean_df[target_variable].values
        figure, ax = plt.subplots(figsize=(20, 10))
        graph = sns.histplot(x, kde=True, color='black')
        plt.axvline(mean, 0, 1, color='red')
        plt.ylabel('# of category')
        plt.xlabel(target_variable)
        # shows values on the top of each bar
        show_values(graph)

        return figure

    def heatmap(self, type_clean_df, target_variable):
        if len(type_clean_df.columns) > 15:
            # Current ratio 15/20
            graph_width = len(type_clean_df.columns) / 1.7
        else:
            graph_width = 13
        correlation = type_clean_df.corr(numeric_only=True)
        matrix = np.triu(correlation)
        colormap = sns.color_palette('flare')
        figure, ax = plt.subplots(figsize=(graph_width, 10))
        hm = sns.heatmap(correlation, annot=True, cmap=colormap, mask=matrix)  # YlGnBu is best color
        for lab in hm.get_yticklabels():
            if lab.get_text() == target_variable:
                lab.set_fontweight('bold')
                lab.set_color('blue')
        for lab in hm.get_xticklabels():
            if lab.get_text() == target_variable:
                lab.set_fontweight('bold')
                lab.set_color('blue')
        plt.xticks(rotation=25)

        return figure

    def unique_value_plot(self, type_clean_df, target_variable):
        # This changes the graph width depending on how many columns the dataframe has
        if len(type_clean_df.columns) > 15:
            graph_width = len(type_clean_df.columns) / 0.5
        else:
            graph_width = 15

        print('graph width:', graph_width)

        figure, ax = plt.subplots(figsize=(graph_width, 10))
        cp = type_clean_df.nunique().plot(kind='bar', color='orange')
        plt.xticks(rotation=25)
        ax.grid(axis='y')

        # This adds min/max/mode labels to the top of each bar
        for column in range(len(type_clean_df.nunique())):
            plt.text(column, type_clean_df.nunique()[column],
                     'Max: ' + str(type_clean_df.max(numeric_only=False)[column]) +
                     '\nMode: ' + str(type_clean_df.mode(numeric_only=False).iloc[0][column]) +
                     '\nMin: ' + str(type_clean_df.min(numeric_only=False)[0]), ha='center', va='bottom')
        # This labels the target variable as blue
        for xtick in cp.get_xticklabels():
            if xtick.get_text() == target_variable:
                xtick.set_fontweight('bold')
                xtick.set_color('blue')

        return figure

    def main_visualizer(self, type_clean_df, target_variable):
        Grapher.box_and_whisker(self, type_clean_df, target_variable)

        def show_graph(event=None):
            try:
                self.full_window_frame2.destroy()
                self.full_window_frame.destroy()

            except:
                pass

            self.full_window_frame = Frame(graphing_window, bg='white')
            self.full_window_frame.pack(fill=BOTH, expand=1)

            self.full_window_canvas = Canvas(self.full_window_frame, bg='white')
            self.full_window_canvas.pack(side=TOP, fill=BOTH, expand=1)

            # Add Scrollbar to canvas
            self.horizontal_scrollbar = ttk.Scrollbar(self.full_window_frame, orient=HORIZONTAL,
                                                      command=self.full_window_canvas.xview)
            self.horizontal_scrollbar.pack(side=BOTTOM, fill=X)

            # Configure the canvas
            self.full_window_canvas.configure(yscrollcommand=self.horizontal_scrollbar.set)
            self.full_window_canvas.bind('<Configure>', lambda e: self.full_window_canvas.configure(
                scrollregion=self.full_window_canvas.bbox('all')))

            # Create a second frame to actually put the graphs into
            self.full_window_frame2 = Frame(self.full_window_canvas, bg='white')

            # add that new frame to a window in canvas
            self.full_window_canvas.create_window((0, 0), window=self.full_window_frame2, anchor='nw')

            if graph_selection_combo_box.get() == 'Target Correlation':
                target_correlation_graph = Grapher.heatmap(self, type_clean_df, target_variable)

                self.selected_graph_canvas = FigureCanvasTkAgg(target_correlation_graph, self.full_window_frame2)
                self.selected_graph_canvas.get_tk_widget().pack(fill=BOTH, expand=1)

            if graph_selection_combo_box.get() == 'All Data Distribution':
                all_data_distribution_graph = Grapher.box_and_whisker(self, type_clean_df, target_variable)

                self.selected_graph_canvas = FigureCanvasTkAgg(all_data_distribution_graph, self.full_window_frame2)
                self.selected_graph_canvas.get_tk_widget().pack(fill=BOTH, expand=1)

            if graph_selection_combo_box.get() == 'Target Distribution':
                target_distribution_graph = Grapher.histogram(self, type_clean_df, target_variable)

                self.selected_graph_canvas = FigureCanvasTkAgg(target_distribution_graph, self.full_window_frame2)
                self.selected_graph_canvas.get_tk_widget().pack(fill=BOTH, expand=1)

            if graph_selection_combo_box.get() == 'Unique Values':
                unique_value_graph = Grapher.unique_value_plot(self, type_clean_df, target_variable)

                self.selected_graph_canvas = FigureCanvasTkAgg(unique_value_graph, self.full_window_frame2)
                self.selected_graph_canvas.get_tk_widget().pack(fill=BOTH, expand=1)

        # Makes sure that the window will only be opened once
        global graphing_window
        try:
            if graphing_window.state() == 'normal': graphing_window.focus()
        except:
            graphing_window = Toplevel()
            graphing_window.title('Data Graphs')
            graphing_window.geometry('800x800')
            graphing_window.state('zoomed')

            # Widgets
            graph_selection_combo_box = ttk.Combobox(graphing_window, values=(
                'Target Correlation', 'Target Distribution', 'All Data Distribution', 'Unique Values'))
            graph_selection_combo_box.current(0)
            graph_selection_combo_box.pack()

            # Bind combo box to select graph
            graph_selection_combo_box.bind('<<ComboboxSelected>>', show_graph)

            show_graph()

            graphing_window.mainloop()


if __name__ == '__main__':
    pass
