import datetime
import pickle
import sqlite3
import warnings
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

import matplotlib.pyplot as plt
import mplcursors as mpc
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.model_selection import KFold


class PredictorTreeviewPage:
    def __init__(self, scaled_df, target_variable, csv_name, split_original_df=None, scaled_df2=None):
        self.scaled_df = scaled_df
        self.target_variable = target_variable
        self.csv_name = csv_name
        self.scaled_df2 = scaled_df2
        self.original_df2 = split_original_df

        # Used for sorting the dataframe
        self.sorted_state = 'off'

        # Checks if a model was selected in the select model window
        try:
            if self.selected_training_model:
                pass
        except AttributeError:
            messagebox.showerror('Error', 'Error: No Model Was Selected')
            return

        # Makes sure only one window is possible to open
        global predictor_window
        try:
            if predictor_window.state() == 'normal': predictor_window.focus()
        except:
            # Create window
            predictor_window = Toplevel(bg='gray10')
            predictor_window.title('Predictor')
            predictor_window.geometry('1100x350')

            # Create frame for treeview
            predictor_treeview_frame = Frame(predictor_window, bg='gray10')
            predictor_treeview_frame.pack(ipadx=200)
            # Create frame for buttons
            self.predictor_buttons_frame = Frame(predictor_window, bg='gray10')
            self.predictor_buttons_frame.pack(ipadx=200)

            # Create tree
            self.predictor_tree = ttk.Treeview(predictor_treeview_frame)

            # Create scrollbar for the tree
            predictor_horizontal_scrollbar = ttk.Scrollbar(predictor_treeview_frame, orient=HORIZONTAL,
                                                           command=self.predictor_tree.xview)
            predictor_horizontal_scrollbar.pack(side=BOTTOM, fill=X)
            self.predictor_tree.configure(xscrollcommand=predictor_horizontal_scrollbar.set)

            # Define columns
            self.predictor_tree['columns'] = (
                'Date_Predicted', 'Target', 'Mean_Absolute_Error', 'Score',
                'Model_Used', 'Dataframe', 'Features_Used', 'All_Train_Df_Predictions',
                'All_Train_Df_Actual_Values')

            # Format columns
            self.predictor_tree.column('#0', width=0, stretch=NO)
            self.predictor_tree.column('Date_Predicted', anchor=W, width=110, stretch=NO)
            self.predictor_tree.column('Target', anchor=W, width=110, stretch=NO)
            self.predictor_tree.column('Mean_Absolute_Error', anchor=W, width=140, stretch=NO)
            self.predictor_tree.column('Score', anchor=W, width=120, stretch=NO)
            self.predictor_tree.column('Model_Used', anchor=W, width=140, stretch=NO)
            self.predictor_tree.column('Dataframe', anchor=W, width=180, stretch=NO)
            self.predictor_tree.column('Features_Used', anchor=W, width=950, stretch=NO)
            self.predictor_tree.column('All_Train_Df_Predictions', anchor=W, width=0, stretch=NO)
            self.predictor_tree.column('All_Train_Df_Actual_Values', anchor=W, width=0, stretch=NO)

            # Create headings
            self.predictor_tree.heading('Date_Predicted', text='Date_Predicted', anchor=W)
            self.predictor_tree.heading('Target', text='Target', anchor=W)
            self.predictor_tree.heading('Mean_Absolute_Error', text='Mean_Absolute_Error', anchor=W)
            self.predictor_tree.heading('Score', text='Score', anchor=W)
            self.predictor_tree.heading('Model_Used', text='Model_Used', anchor=W)
            self.predictor_tree.heading('Dataframe', text='Dataframe', anchor=W)
            self.predictor_tree.heading('Features_Used', text='Features_Used', anchor=W)
            self.predictor_tree.heading('All_Train_Df_Predictions', text='All_Train_Df_Predictions', anchor=W)
            self.predictor_tree.heading('All_Train_Df_Actual_Values', text='All_Train_Df_Actual_Values', anchor=W)

            # Bind treeview to click for filter
            self.predictor_tree.bind('<Button-1>', self.on_column_clicked)

            # Widgets
            create_new_prediction_button = ttk.Button(self.predictor_buttons_frame,
                                                      text='Create New Predictions For Train Dataframe',
                                                      command=self.on_create_new_prediction)
            delete_selected_row_button = ttk.Button(self.predictor_buttons_frame, text='Delete Selected Line',
                                                    command=self.on_delete_selected_row)
            up_button = ttk.Button(self.predictor_buttons_frame, text='Move Current Line Up',
                                   command=self.on_move_current_line_up)
            down_button = ttk.Button(self.predictor_buttons_frame, text='Move Current Line Down',
                                     command=self.on_move_current_line_down)
            row_order_saver_button = ttk.Button(self.predictor_buttons_frame, text='Save Current Row Order',
                                                command=self.on_save_current_row_order)
            self.filter_current_model_checkbox = ttk.Checkbutton(self.predictor_buttons_frame,
                                                                 text='Only Show Current Model',
                                                                 command=self.filter_model)
            show_current_lines_error_graph_button = ttk.Button(self.predictor_buttons_frame,
                                                               text="Show Current Line's Error Graph",
                                                               command=self.on_show_current_lines_error_graph)
            view_train_df_predictions_button = ttk.Button(self.predictor_buttons_frame,
                                                          text='View All Predictions For Train Dataframe',
                                                          command=self.on_view_train_df_predictions)
            view_test_df_predictions_button = ttk.Button(self.predictor_buttons_frame,
                                                         text='View All Predictions For Test Dataframe',
                                                         command=self.on_view_test_df_predictions)

            # Makes the current dataframe checkbox start out as selected
            self.filter_current_model_checkbox.state(['!alternate'])
            self.filter_current_model_checkbox.state(['selected'])

            # Locations
            self.predictor_tree.pack(expand=True, fill=BOTH, padx=5)

            create_new_prediction_button.grid(row=0, column=0, padx=20, pady=5)
            view_train_df_predictions_button.grid(row=1, column=0, padx=20, pady=5)
            view_test_df_predictions_button.grid(row=2, column=0, padx=20, pady=5)

            delete_selected_row_button.grid(row=1, column=1, padx=20, pady=5)
            self.filter_current_model_checkbox.grid(row=0, column=1, padx=20, pady=5)
            show_current_lines_error_graph_button.grid(row=2, column=1, padx=20, pady=5)

            up_button.grid(row=0, column=4, padx=20, pady=5)
            down_button.grid(row=1, column=4, padx=20, pady=5)
            row_order_saver_button.grid(row=2, column=4, padx=20, pady=5)

            self.query_database()
            predictor_window.mainloop()

    def filter_model(self):
        if self.filter_current_model_checkbox.instate(['selected']) == True:
            pass
        else:
            # Clear the treeview table
            self.predictor_tree.delete(*self.predictor_tree.get_children())
            self.query_database()
            return

        # creates a database if one doesn't already exist, otherwise it connects to it
        conn = sqlite3.connect(
            '../../Databases/Predictions_Database')  # creates a database file and puts it in the directory

        # creates a cursor that does all the editing
        cursor = conn.cursor()

        # Grab only data you want
        cursor.execute('SELECT * FROM predictions_table WHERE Model_Used = :Model_Used',
                       {'Model_Used': self.selected_training_model})
        fetched_records = cursor.fetchall()

        conn.commit()
        conn.close()

        # Clear the treeview table
        self.predictor_tree.delete(*self.predictor_tree.get_children())

        # Add new data to the screen
        count = 0
        for record in fetched_records:
            self.predictor_tree.insert(parent='', index='end', iid=count, text='',
                                       values=(record[0], record[1], record[2], record[3], record[4],
                                               record[5], record[6], record[7], record[8]))
            # Increment counter
            count += 1

    def on_column_clicked(self, event):
        region_clicked = self.predictor_tree.identify_region(event.x, event.y)

        if region_clicked not in 'heading':
            return
        if self.sorted_state == 'off':
            column_clicked = self.predictor_tree.identify_column(event.x)
            column_clicked_index = int(column_clicked[1:]) - 1

            self.sorted_state = 'on'
            column_clicked_name = (self.predictor_tree['columns'][column_clicked_index])

            # Puts a down arrow in the column name
            self.predictor_tree.heading(column_clicked_name, text=column_clicked_name + ' ' * 3 + 'V')

            conn = sqlite3.connect(
                '../../Databases/Predictions_Database')  # creates a database file and puts it in the directory

            # creates a cursor that does all the editing
            cursor = conn.cursor()

            # query the database (which means save) addresses=the table, oid = unique original id number
            cursor.execute('SELECT *, oid FROM predictions_table ORDER BY ' + column_clicked_name + ' DESC')
            fetched_records = cursor.fetchall()  # fetchone() brings back one record, fetchmany() brings many, fetchall() brings all records

            # Commit changes
            conn.commit()
            # Close connection
            conn.close()

            # Clear the treeview table
            self.predictor_tree.delete(*self.predictor_tree.get_children())

            # Refill the treeview table
            count = 0
            for record in fetched_records:
                self.predictor_tree.insert(parent='', index='end', iid=count, text='',
                                           values=(
                                               record[0], record[1], record[2], record[3], record[4], record[5],
                                               record[6],
                                               record[7], record[8]))
                # Increment counter
                count += 1

        else:
            # Reload the original treeview data
            for column in self.predictor_tree['columns']:
                self.predictor_tree.heading(column, text=column)  # Reload the original treeview data

            self.sorted_state = 'off'

    def on_create_new_prediction(self):
        def prediction_database_inserter():
            self.all_predictions, all_actual_values = Predictor.train_predictor(self, self.scaled_df)

            # Scale the predictions back to normal
            # Back-transform the target variable column
            if self.target_skew_winner == 'log transformer':
                all_actual_values = [np.exp(value) for value in all_actual_values]
                self.all_predictions = [np.exp(prediction) for prediction in self.all_predictions]
            if self.target_skew_winner == 'sqr root transformer':
                all_actual_values = [value ** 2 for value in all_actual_values]
                self.all_predictions = [prediction ** 2 for prediction in self.all_predictions]
            if self.target_skew_winner == 'box cox transformer':
                from scipy.special import inv_boxcox
                all_actual_values = [inv_boxcox(value) for values in all_actual_values]
                self.all_predictions = [inv_boxcox(prediction) for prediction in self.all_predictions]

            # Convert the above to usable sql format
            # all_actual_values = all_actual_values.tolist()
            self.all_predictions = ', '.join(map(str, self.all_predictions))
            all_actual_values = ', '.join(map(str, all_actual_values))

            selected_features = self.selected_features
            selected_features.remove(self.target_variable)
            database_feature_combination = ', '.join(selected_features)

            # Connect to database
            conn = sqlite3.connect('../../Databases/Predictions_Database')

            # Create cursor
            cursor = conn.cursor()

            # Add new record
            cursor.execute(
                "INSERT INTO predictions_table VALUES (:Date_Predicted, :Target, :Mean_Absolute_Error, :Score, :Model_Used, :Dataframe, :Features_Used, :All_Predictions, :All_Actual_Values)",
                {'Date_Predicted': datetime.date.today(),
                 'Target': self.target_variable,
                 'Mean_Absolute_Error': self.train_average_mae,
                 'Score': round(self.train_average_score, 15),
                 'Model_Used': self.selected_training_model,
                 'Dataframe': str(self.csv_name),
                 'Features_Used': database_feature_combination,
                 'All_Predictions': self.all_predictions,
                 'All_Actual_Values': all_actual_values})

            # Commit changes
            conn.commit()

            # Close connection
            conn.close()

            # Clear the treeview table
            self.predictor_tree.delete(*self.predictor_tree.get_children())

            # Reset tree by querying the database again
            self.query_database()

        prediction_database_inserter()

    def on_delete_selected_row(self):
        selected = self.predictor_tree.selection()[0]
        tree_values = self.predictor_tree.item(selected, 'values')

        # Create a database or connect to one that already exists
        conn = sqlite3.connect('../../Databases/Predictions_Database')

        # Create a cursor instance
        cursor = conn.cursor()

        # Delete from database
        cursor.execute(
            'DELETE from predictions_table WHERE Date_Predicted = :Date_Predicted AND Target = :Target AND Dataframe = :Dataframe AND Model_Used = :Model_Used AND Score = :Score AND Features_Used = :Features_Used',
            {'Date_Predicted': tree_values[0],
             'Target': tree_values[1],
             'Score': tree_values[3],
             'Model_Used': tree_values[4],
             'Dataframe': tree_values[5],
             'Features_Used': tree_values[6]})
        # Commit changes
        conn.commit()

        # Close our connection
        conn.close()

        # Clear the treeview table
        self.predictor_tree.delete(*self.predictor_tree.get_children())

        # Reset tree by querying the database again
        self.query_database()

        row_deleted_label = Label(self.predictor_buttons_frame, text='Selected Row Removed', fg='red')
        row_deleted_label.grid(row=3, column=1)

    def on_show_current_lines_error_graph(self):
        # Get all predicted and test values from treeview (these are hidden and are not displayed in the treeview)
        selected = self.predictor_tree.selection()[0]
        tree_all_predicted_values = self.predictor_tree.item(selected, 'values')[7]
        tree_all_actual_values = self.predictor_tree.item(selected, 'values')[8]

        # Split long string from treeview into a list
        tree_all_predicted_values = tree_all_predicted_values.split(', ')
        tree_all_actual_values = tree_all_actual_values.split(', ')

        # Turn tree values into arrays for plotter
        tree_all_predicted_values = np.array([float(value) for value in tree_all_predicted_values])
        tree_all_actual_values = np.array([float(value) for value in tree_all_actual_values])

        # Run plotter
        Predictor.predictor_plotter(self, tree_all_predicted_values, tree_all_actual_values, self.target_variable)

    def on_move_current_line_down(self):
        rows = self.predictor_tree.selection()
        for row in reversed(rows):
            self.predictor_tree.move(row, self.predictor_tree.parent(row), self.predictor_tree.index(row) + 1)

    def on_move_current_line_up(self):
        rows = self.predictor_tree.selection()
        for row in rows:
            self.predictor_tree.move(row, self.predictor_tree.parent(row), self.predictor_tree.index(row) - 1)

    def on_save_current_row_order(self):
        # Connect to database
        conn = sqlite3.connect('../../Databases/Predictions_Database')

        # Create cursor
        cursor = conn.cursor()

        # Delete old data order
        cursor.execute('DELETE FROM predictions_table')

        # Commit changes
        conn.commit()

        # Close connection
        conn.close()

        # Add new record
        for record in self.predictor_tree.get_children():
            # Connect to database
            conn = sqlite3.connect('../../Databases/Predictions_Database')

            # Create cursor
            cursor = conn.cursor()

            # Insert new reordered data into table
            cursor.execute(
                "INSERT INTO predictions_table VALUES (:Date_Predicted, :Target, :Mean_Absolute_Error, :Score, :Model_Used, :Dataframe, :Features_Used, :All_Train_Df_Predictions, :All_Train_Df_Actual_Values)",
                {'Date_Predicted': self.predictor_tree.item(record[0], 'values')[0],
                 'Target': self.predictor_tree.item(record[0], 'values')[1],
                 'Mean_Absolute_Error': self.predictor_tree.item(record[0], 'values')[2],
                 'Score': self.predictor_tree.item(record[0], 'values')[3],
                 'Model_Used': self.predictor_tree.item(record[0], 'values')[4],
                 'Dataframe': self.predictor_tree.item(record[0], 'values')[5],
                 'Features_Used': self.predictor_tree.item(record[0], 'values')[6],
                 'All_Train_Df_Predictions': self.predictor_tree.item(record[0], 'values')[7],
                 'All_Train_Df_Actual_Values': self.predictor_tree.item(record[0], 'values')[8]})

            # Commit changes
            conn.commit()

            # Close connection
            conn.close()

        # Clear the treeview table
        self.predictor_tree.delete(*self.predictor_tree.get_children())

        # Refresh the database
        self.query_database()

    def on_view_test_df_predictions(self):
        def csv_name_entry_initial_clearer(e, csv_name_entry):
            csv_name_entry.delete(0, END)

        def on_column_clicked__test():
            pass

        def on_save_to_csv(predicted_df, test_df_predictions_button_frame):
            def on_save_button():
                predicted_df.to_csv('../../Predictions/' + csv_name_entry.get() + '.csv', index=False, encoding='utf-8')

                file_saved_label = Label(test_df_predictions_button_frame, text='File Saved Successfully', fg='green')
                file_saved_label.grid(row=3, column=0, pady=5)

            # Create entry box for csv name
            csv_name_entry = Entry(test_df_predictions_button_frame, width=17, font=('Arial', 11, 'italic'))
            csv_name_entry.insert(0, 'Save As')
            save_button = ttk.Button(test_df_predictions_button_frame, text='Save', command=on_save_button)

            # Make entry box clear when clicked on
            csv_name_entry.bind('<ButtonRelease-1>',
                                lambda event, csv_name_entry=csv_name_entry: csv_name_entry_initial_clearer(event,
                                                                                                            csv_name_entry))
            # ToDo put in vertical scrollbar for predictions treeview
            # Locations
            csv_name_entry.grid(row=1, column=0, padx=5)
            save_button.grid(row=2, column=0)

        # Makes sure only one window is possible to open
        global test_df_predictions_window
        try:
            if test_df_predictions_window.state() == 'normal': test_df_predictions_window.focus()
        except:
            # Create window
            test_df_predictions_window = Toplevel()
            test_df_predictions_window.title('Test Dataframe Predictions')
            test_df_predictions_window.geometry('600x380')

            # Create frame for treeview
            test_df_predictions_tree_frame = Frame(test_df_predictions_window)
            test_df_predictions_tree_frame.pack(ipadx=200)

            # Create frame for buttons
            test_df_predictions_button_frame = Frame(test_df_predictions_window)
            test_df_predictions_button_frame.pack(ipadx=200)

            # Create tree
            test_df_predictions_tree = ttk.Treeview(test_df_predictions_tree_frame)

            # Create scrollbar for tree frame
            test_df_predictions_scrollbar = ttk.Scrollbar(test_df_predictions_tree_frame, orient=HORIZONTAL,
                                                          command=test_df_predictions_tree.xview)
            test_df_predictions_scrollbar.pack(side=BOTTOM, fill=X)
            test_df_predictions_tree.configure(xscrollcommand=test_df_predictions_scrollbar.set)

            # Get average predictions
            test_average_predictions = Predictor.test_predictor(self, self.scaled_df, self.scaled_df2)
            test_average_predictions_df = pd.DataFrame(test_average_predictions, columns=['Predicted_Value'])

            # Combine the predicted values with the split original dataframe
            test_predicted_df = pd.concat([self.original_df2, test_average_predictions_df], axis=1)

            # Back-transform the target variable column
            if self.target_skew_winner == 'log transformer':
                test_predicted_df['Predicted_Value'] = np.exp(test_predicted_df['Predicted_Value'])
            if self.target_skew_winner == 'sqr root transformer':
                test_predicted_df['Predicted_Value'] = test_predicted_df['Predicted_Value'] ** 2
            if self.target_skew_winner == 'box cox transformer':
                from scipy.special import inv_boxcox
                test_predicted_df['Predicted_Value'] = inv_boxcox(test_predicted_df['Predicted_Value'],
                                                                  self.target_box_cox_lambda)

            # Define columns
            test_df_predictions_tree['columns'] = test_predicted_df.columns.tolist()
            test_df_predictions_tree['show'] = 'headings'

            # Loop through column list to create the tree headers
            for column in test_df_predictions_tree['columns']:
                test_df_predictions_tree.heading(column, text=column, anchor=W)

            # Put data in treeview
            df_rows = test_predicted_df.to_numpy().tolist()
            for row in df_rows:
                test_df_predictions_tree.insert('', 'end', values=row)

            # Bind treeview to column click for filter
            test_df_predictions_tree.bind('<Button-1>', on_column_clicked__test)

            # Widgets
            save_to_csv_button__test = ttk.Button(test_df_predictions_button_frame, text='Save to CSV',
                                                  command=lambda: on_save_to_csv(test_predicted_df,
                                                                                 test_df_predictions_button_frame))

            # ToDo put all of this parts data into treeview hidden instead of having to create new prediction to create

            # Locations
            test_df_predictions_tree.pack(expand=True, fill=BOTH, padx=5)
            save_to_csv_button__test.grid(row=0, column=0, padx=5, pady=5)

    def on_view_train_df_predictions(self):
        def csv_name_entry_initial_clearer(e, csv_name_entry):
            csv_name_entry.delete(0, END)

        def on_column_clicked__train():
            pass

        def on_save_to_csv(train_df_predictions_button_frame, tree_all_predicted_values_list,
                           tree_all_actual_values_list):

            def on_save_button():
                train_df_predictions.to_csv('../../Predictions/' + csv_name_entry.get() + '.csv', index=False,
                                            encoding='utf-8')

                file_saved_label = Label(train_df_predictions_button_frame, text='File Saved Successfully', fg='green')
                file_saved_label.grid(row=3, column=0, pady=5)

            differences_list = []
            for row in range(len(tree_all_predicted_values_list)):
                differences_list.append(
                    float(tree_all_predicted_values_list[row]) - float(tree_all_actual_values_list[row]))

            train_file_data = list(zip(tree_all_predicted_values_list, tree_all_actual_values_list, differences_list))

            train_df_predictions = pd.DataFrame(train_file_data,
                                                columns=['Predicted Values', 'Actual Values', 'Difference'])
            # Create entry box for csv name
            csv_name_entry = Entry(train_df_predictions_button_frame, width=17, font=('Arial', 11, 'italic'))
            csv_name_entry.insert(0, 'Save As')
            save_button = ttk.Button(train_df_predictions_button_frame, text='Save', command=on_save_button)

            # Make entry box clear when clicked on
            csv_name_entry.bind('<ButtonRelease-1>',
                                lambda event, csv_name_entry=csv_name_entry: csv_name_entry_initial_clearer(event,
                                                                                                            csv_name_entry))

            # Locations
            csv_name_entry.grid(row=1, column=0, padx=5)
            save_button.grid(row=2, column=0)

        # Makes sure only one window is possible to open
        global train_df_predictions_window
        try:
            if train_df_predictions_window.state() == 'normal': train_df_predictions_window.focus()
        except:
            # Create window
            train_df_predictions_window = Toplevel()
            train_df_predictions_window.title('Train Dataframe Predictions')
            train_df_predictions_window.geometry('400x350')

            # Create frame for treeview
            train_df_predictions_tree_frame = Frame(train_df_predictions_window)
            train_df_predictions_tree_frame.pack(ipadx=200)

            # Create frame for buttons
            train_df_predictions_button_frame = Frame(train_df_predictions_window)
            train_df_predictions_button_frame.pack(ipadx=200)

            # Create tree
            train_df_predictions_tree = ttk.Treeview(train_df_predictions_tree_frame)

            # Define columns
            train_df_predictions_tree['columns'] = ('Predicted_Value', 'Actual_Value', 'Difference')

            # Format columns
            train_df_predictions_tree.column('#0', width=0, stretch=NO)
            train_df_predictions_tree.column('Predicted_Value', width=140, stretch=NO)
            train_df_predictions_tree.column('Actual_Value', width=120, stretch=NO)
            train_df_predictions_tree.column('Difference', width=120, stretch=NO)

            # Create headings
            train_df_predictions_tree.heading('Predicted_Value', text='Predicted_Value', anchor=W)
            train_df_predictions_tree.heading('Actual_Value', text='Actual_Value', anchor=W)
            train_df_predictions_tree.heading('Difference', text='Difference', anchor=W)

            # Bind treeview to column click for filter
            train_df_predictions_tree.bind('<Button-1>', on_column_clicked__train)

            # Grab values from predictor treeview current selected
            selected = self.predictor_tree.selection()
            train_df_all_predicted_values = self.predictor_tree.item(selected, 'values')[7]
            train_df_all_actual_values = self.predictor_tree.item(selected, 'values')[8]

            train_df_all_predicted_values_list = train_df_all_predicted_values.split(', ')
            train_df_all_actual_values_list = train_df_all_actual_values.split(', ')

            # Create an iteration to grab data and put it into treeview
            index = max_difference = total_difference = 0
            min_difference = 99999999999999999999999999
            for row in range(len(train_df_all_predicted_values_list)):
                train_df_predictions_tree.insert(parent='', index='end', iid=index, text='',
                                                 values=(str(train_df_all_predicted_values_list[row]),
                                                         str(train_df_all_actual_values_list[row]),
                                                         (float(train_df_all_predicted_values_list[row]) - float(
                                                             train_df_all_actual_values_list[row]))))

                individual_difference = abs(
                    float(train_df_all_predicted_values_list[row]) - float(train_df_all_actual_values_list[row]))
                total_difference = total_difference + individual_difference

                if individual_difference > max_difference:
                    max_difference = individual_difference
                if individual_difference < min_difference:
                    min_difference = individual_difference
                index += 1

            # Widgets
            save_to_csv_button__train = ttk.Button(train_df_predictions_button_frame, text='Save to CSV',
                                                   command=lambda: on_save_to_csv(train_df_predictions_button_frame,
                                                                                  train_df_all_predicted_values_list,

                                                                                  train_df_all_actual_values_list))

            avg_difference_label__train = Label(train_df_predictions_button_frame,
                                                text='Average Difference: ' + str(total_difference / (index + 1)))
            max_difference_label__train = Label(train_df_predictions_button_frame,
                                                text='Max Difference: ' + str(max_difference))
            min_difference_label__train = Label(train_df_predictions_button_frame,
                                                text='Minimum Difference: ' + str(min_difference))

            # Locations
            train_df_predictions_tree.pack(expand=True, fill=BOTH, padx=5)
            save_to_csv_button__train.grid(row=0, column=0, padx=5, pady=5)
            avg_difference_label__train.grid(row=0, column=1, padx=5, pady=5)
            max_difference_label__train.grid(row=1, column=1, padx=5, pady=5)
            min_difference_label__train.grid(row=2, column=1, padx=5, pady=5)

    def query_database(self):
        if self.filter_current_model_checkbox.instate(['selected']) == True:
            self.filter_model()
            return
        else:
            pass

        # creates a database if one doesn't already exist, otherwise it connects to it
        conn = sqlite3.connect(
            '../../Databases/Predictions_Database')  # creates a database file and puts it in the directory

        # creates a cursor that does all the editing
        cursor = conn.cursor()

        # Create table in database if it doesn't already exist
        cursor.execute("""CREATE TABLE if not exists predictions_table (
                Date_Predicted DATE,
                Target text,
                Mean_Absolute_Error real,
                Score real,
                Model_Used text,
                Dataframe text,
                Features_Used text,
                All_Train_Df_Predictions text,
                All_Train_Df_Actual_Values text
                )""")

        # query the database (which means save) addresses=the table, oid = unique original id number
        cursor.execute('SELECT *, oid FROM predictions_table')
        fetched_records = cursor.fetchall()  # fetchone() brings back one record, fetchmany() brings many, fetchall() brings all records

        # Commit changes
        conn.commit()
        # Close connection
        conn.close()

        # Clear the treeview table
        self.predictor_tree.delete(*self.predictor_tree.get_children())

        # Add our data to the screen (first time only)
        global count
        count = 0
        for record in fetched_records:
            self.predictor_tree.insert(parent='', index='end', iid=count, text='',
                                       values=(record[0], record[1], record[2], record[3], record[4], record[5],
                                               record[6], record[7], record[8]))
            # Increment counter
            count += 1


class Predictor:
    # This will get the mean values for each column of the full_df,
    # then it will add entire columns all with mean values to the shortened_df
    def predictor_array_cleaner(self, full_df, shortened_df):
        # Grabs the mean values of each full_df column
        full_df_mean = pd.DataFrame(full_df.mean())

        # Switches the rows and columns
        full_df_mean = full_df_mean.T

        # Creates columns for every column that is not in the shortened_df and sets it to the mean of full_df's entire column
        for column in full_df_mean.columns:
            if column not in shortened_df.columns:
                shortened_df[column] = full_df_mean[column][0]

        # Now turn the new dataframe into an array
        finalized_predictor_array = np.array(shortened_df)
        return finalized_predictor_array

    def train_predictor(self, dataframe):
        # Turn off warnings
        warnings.filterwarnings('ignore')

        selected_features = self.selected_features
        selected_features.append(self.target_variable)

        # Make dataframe from selected features
        shortened_dataframe = dataframe[self.selected_features]

        X = np.array(shortened_dataframe.drop([self.target_variable], axis=1), dtype='object')
        y = np.array(shortened_dataframe[self.target_variable], dtype='object')

        # ToDo put in a quick predictor? Using the line below
        # finalized_predictor_array = Predictor.predictor_array_cleaner(self, shortened_dataframe, self.target_variable)
        pickle_in = open('../../saved_models/' + self.selected_training_model + '.pickle', 'rb')
        pickled_weights_and_models_dict = pickle.load(pickle_in)
        pickle_in.close()
        models = pickled_weights_and_models_dict.keys()
        weights = pickled_weights_and_models_dict.values()

        runtimes = 100

        total_predictions = total_score = total_mean_absolute_error = 0
        for i in range(runtimes):
            self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                                            test_size=0.2)

            results_dict = {}
            kf = KFold(n_splits=10)

            for model in pickled_weights_and_models_dict.keys():
                model.fit(X, y)
                # score = cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
                # results_dict[name] = score

            predictions = (
                    list(models)[0].predict(self.X_test) * list(weights)[0] +
                    list(models)[1].predict(self.X_test) * list(weights)[1] +
                    list(models)[2].predict(self.X_test) * list(weights)[2] +
                    list(models)[3].predict(self.X_test) * list(weights)[3] +
                    list(models)[4].predict(self.X_test) * list(weights)[4] +
                    list(models)[5].predict(self.X_test) * list(weights)[5]
            )

            # score = (
            #     cross_val_score(list(model)[0], X, y, cv=kf, scoring='r2').mean() * list(weights)[0] +
            #     cross_val_score(list(model)[1], X, y, cv=kf, scoring='r2').mean() * list(weights)[0] +
            #     cross_val_score(list(model)[2], X, y, cv=kf, scoring='r2').mean() * list(weights)[0] +
            #     cross_val_score(list(model)[3], X, y, cv=kf, scoring='r2').mean() * list(weights)[0] +
            #     cross_val_score(list(model)[4], X, y, cv=kf, scoring='r2').mean() * list(weights)[0] +
            #     cross_val_score(list(model)[5], X, y, cv=kf, scoring='r2').mean() * list(weight)[0]
            #     )

            score = (
                    list(models)[0].score(self.X_test, self.y_test) * list(weights)[0] +
                    list(models)[1].score(self.X_test, self.y_test) * list(weights)[1] +
                    list(models)[2].score(self.X_test, self.y_test) * list(weights)[2] +
                    list(models)[3].score(self.X_test, self.y_test) * list(weights)[3] +
                    list(models)[4].score(self.X_test, self.y_test) * list(weights)[4] +
                    list(models)[5].score(self.X_test, self.y_test) * list(weights)[5]
            )

            # predictions = regression_line.predict(self.X_test)
            # score = regression_line.score(self.X_test, self.y_test)
            mean_absolute_error = metrics.mean_absolute_error(self.y_test, predictions)

            total_predictions = np.add(predictions, total_predictions)
            total_score += score
            total_mean_absolute_error += mean_absolute_error

            # ToDo make sure raw doesn't win everytime in the future

        self.train_average_predictions = total_predictions / runtimes

        self.train_average_score = total_score / runtimes
        self.train_average_mae = total_mean_absolute_error / runtimes

        self.train_all_actual_values = self.y_test

        for model in models:
            try:
                model_and_coefs_list = list(zip(self.selected_features, model.coef_))
                sorted_model_and_coefs_list = reversed(sorted(model_and_coefs_list, key=lambda x: x[1]))

                total = 0
                for coef in sorted_model_and_coefs_list:
                    total += round(coef[1], 3)
            except:
                continue

        return self.train_average_predictions, self.train_all_actual_values

    def test_predictor(self, scaled_df, scaled_df2):
        # Loads in the regression line using pickle
        pickle_in = open('../../Saved_Models/' + self.selected_training_model + '.pickle', 'rb')
        pickled_weights_and_models_dict = pickle.load(pickle_in)
        pickle_in.close()
        models = pickled_weights_and_models_dict.keys()
        weights = pickled_weights_and_models_dict.values()

        scaled_df__selected = scaled_df[self.selected_features]
        scaled_df2__selected = scaled_df2[self.selected_features]
        finalized_predictor_array = Predictor.predictor_array_cleaner(self, scaled_df__selected, scaled_df2__selected)

        runtimes = 10000
        total_predictions = 0
        for i in range(runtimes):
            predictions = (
                    list(models)[0].predict(finalized_predictor_array) * list(weights)[0] +
                    list(models)[1].predict(finalized_predictor_array) * list(weights)[1] +
                    list(models)[2].predict(finalized_predictor_array) * list(weights)[2] +
                    list(models)[3].predict(finalized_predictor_array) * list(weights)[3] +
                    list(models)[4].predict(finalized_predictor_array) * list(weights)[4] +
                    list(models)[5].predict(finalized_predictor_array) * list(weights)[5]
            )

            # predictions = regression_line.predict(finalized_predictor_array)
            total_predictions = np.add(predictions, total_predictions)
        test_average_predictions = total_predictions / runtimes

        return test_average_predictions

    def predictor_plotter(self, tree_all_predicted_values, tree_all_actual_values, target_variable):
        # Sort and combine tree all predicted values and tree all actual values
        prediction_plotter_df = pd.DataFrame()
        prediction_plotter_df['tree_all_actual_values'] = tree_all_actual_values
        prediction_plotter_df['tree_all_predicted_values'] = tree_all_predicted_values
        prediction_plotter_df = prediction_plotter_df.sort_values(by=['tree_all_actual_values'])

        sns.set_style('darkgrid')
        plt.figure(figsize=(15, 10))
        # Plot all actual values
        plt.plot(prediction_plotter_df['tree_all_actual_values'], (
                prediction_plotter_df['tree_all_actual_values'] - prediction_plotter_df[
            'tree_all_predicted_values']).abs())
        plt.xlabel('Actual Values for ' + self.target_variable)
        plt.ylabel('Error for Each Input')
        plt.title('Actual Values * Error for Each Input')
        plt.get_current_fig_manager().window.state('zoomed')

        # Makes the point values shown when hovered over
        mpc.cursor(hover=True)

        plt.show()


if __name__ == '__main__':
    print('You must run this program from the mother predictor app')
