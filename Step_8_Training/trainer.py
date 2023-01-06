import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import time
from Extras import functions
import pandas as pd
import concurrent.futures
import ast
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import sqlite3
import datetime
import os


class TrainingModelPage:
    def __init__(self, scaled_df, target_variable, csv_name):
        self.scaled_df = scaled_df
        self.target_variable = target_variable
        self.csv_name = csv_name

        # Used for sorting the dataframe
        self.sorted_state = 'off'

        # Checks if any features were selected in the select features window
        try:
            if self.selected_features:
                pass
        except AttributeError:
            messagebox.showerror('Error', 'Error: No Features Were Selected')
            return

        # Makes sure only one window at at time will open
        global training_model_window
        try:
            if training_model_window.state() == 'normal': training_model_window.focus()
        except:
            # Create window
            training_model_window = Toplevel()
            training_model_window.title('Training Models')
            training_model_window.geometry('1100x350')

            # Create frame for tree
            training_model_tree_frame = Frame(training_model_window)
            training_model_tree_frame.pack(ipadx=200)
            # Create frame for buttons
            self.training_model_button_frame = Frame(training_model_window)
            self.training_model_button_frame.pack(ipadx=200)

            # Create tree
            self.training_model_tree = ttk.Treeview(training_model_tree_frame)

            # Configure scrollbar
            training_horizontal_scrollbar = ttk.Scrollbar(training_model_tree_frame, orient=HORIZONTAL,
                                                          command=self.training_model_tree.xview)
            training_horizontal_scrollbar.pack(side=BOTTOM, fill=X)
            self.training_model_tree.configure(xscrollcommand=training_horizontal_scrollbar.set)

            # Define columns
            self.training_model_tree['columns'] = (
                'Date_Created', 'Target', 'Dataframe', 'Saved_Model_Name', 'Total_Average_Score',
                'Total_Model_Upgrades', 'Last_Upgrade_Total_Runtimes', 'Total_Runtimes', 'Best_Average_Score',
                'Best_Score_Runtimes', 'Features_Used')

            # Format columns
            self.training_model_tree.column('#0', width=0, stretch=NO)
            self.training_model_tree.column('Date_Created', anchor=W, width=100, stretch=NO)
            self.training_model_tree.column('Target', anchor=W, width=130, stretch=NO)
            self.training_model_tree.column('Dataframe', anchor=W, width=180, stretch=NO)
            self.training_model_tree.column('Saved_Model_Name', anchor=W, width=240, stretch=NO)
            self.training_model_tree.column('Total_Average_Score', anchor=W, width=150, stretch=NO)
            self.training_model_tree.column('Total_Model_Upgrades', anchor=W, width=150, stretch=NO)
            self.training_model_tree.column('Last_Upgrade_Total_Runtimes', anchor=W, width=170, stretch=NO)
            self.training_model_tree.column('Total_Runtimes', anchor=W, width=120, stretch=NO)
            self.training_model_tree.column('Best_Average_Score', anchor=W, width=155, stretch=NO)
            self.training_model_tree.column('Best_Score_Runtimes', anchor=W, width=160, stretch=NO)
            self.training_model_tree.column('Features_Used', anchor=W, width=1000, stretch=NO)

            # ToDo make total average score weighted by the runtimes amount. Maybe it already is?

            # Create headings
            self.training_model_tree.heading('Date_Created', text='Date_Created', anchor=W)
            self.training_model_tree.heading('Target', text='Target', anchor=W)
            self.training_model_tree.heading('Dataframe', text='Dataframe', anchor=W)
            self.training_model_tree.heading('Saved_Model_Name', text='Saved_Model_Name', anchor=W)
            self.training_model_tree.heading('Total_Average_Score', text='Total Average_Score', anchor=W)
            self.training_model_tree.heading('Total_Model_Upgrades', text='Total_Model_Upgrades', anchor=W)
            self.training_model_tree.heading('Last_Upgrade_Total_Runtimes', text='Last_Upgrade_Total_Runtimes',
                                             anchor=W)
            self.training_model_tree.heading('Total_Runtimes', text='Total_Runtimes', anchor=W)
            self.training_model_tree.heading('Best_Average_Score', text='Best_Average_Score', anchor=W)
            self.training_model_tree.heading('Best_Score_Runtimes', text='Best_Score_Runtimes', anchor=W)
            self.training_model_tree.heading('Features_Used', text='Features_Used', anchor=W)

            # Bind treeview to click
            self.training_model_tree.bind('<Double-Button-1>', self.on_use_selected_model)
            self.training_model_tree.bind('<Button-1>', self.on_column_clicked)

            # Widgets
            use_selected_training_model_button = ttk.Button(self.training_model_button_frame, text='Use Selected Model',
                                                            command=self.on_use_selected_model)
            create_new_training_model_button = ttk.Button(self.training_model_button_frame, text='Create New Model',
                                                          command=lambda: Trainer.new_model_creator(self,
                                                                                                    self.selected_features,
                                                                                                    self.scaled_df,
                                                                                                    self.target_variable))
            delete_selected_model_button = ttk.Button(self.training_model_button_frame, text='Delete Selected Model',
                                                      command=self.on_delete_selected_model)
            up_button = ttk.Button(self.training_model_button_frame, text='Move Record Up',
                                   command=self.on_move_record_up)
            down_button = ttk.Button(self.training_model_button_frame, text='Move Record Down',
                                     command=self.on_move_record_down)
            further_training_button = ttk.Button(self.training_model_button_frame, text='Further Train Selected Model',
                                                 command=self.on_further_train)
            row_order_saver_button = ttk.Button(self.training_model_button_frame, text='Save Current Row Order',
                                                command=self.on_save_current_row_order)
            self.filter_current_features_checkbox = ttk.Checkbutton(self.training_model_button_frame,
                                                                    text='Only Show Current Target and Selected Features',
                                                                    command=self.filter_features)

            # Makes the current features checkbox start out as selected
            self.filter_current_features_checkbox.state(['!alternate'])
            self.filter_current_features_checkbox.state(['selected'])

            # Locations
            self.training_model_tree.pack(expand=True, fill=BOTH, padx=5)
            use_selected_training_model_button.grid(row=1, column=0)
            create_new_training_model_button.grid(row=1, column=1)
            delete_selected_model_button.grid(row=3, column=1)
            up_button.grid(row=2, column=0)
            down_button.grid(row=3, column=0)
            further_training_button.grid(row=2, column=1)
            row_order_saver_button.grid(row=4, column=1)
            self.filter_current_features_checkbox.grid(row=1, column=2)

            self.query_database()

    def filter_features(self):
        if self.filter_current_features_checkbox.instate(['selected']) == True:
            pass
        else:
            # Clear the treeview table
            self.training_model_tree.delete(*self.training_model_tree.get_children())
            self.query_database()
            return

        # Change self.selected features into a database usable format
        selected_features = self.selected_features
        selected_features = ', '.join(selected_features)

        # creates a database if one doesn't already exist, otherwise it connects to it
        conn = sqlite3.connect('Training_Model_Database')  # creates a database file and puts it in the directory

        # creates a cursor that does all the editing
        cursor = conn.cursor()


        # Grab only data you want
        cursor.execute('SELECT * FROM training_model_table WHERE Target = :Target AND Features_Used = :Features_Used',
                       {'Target': self.target_variable,
                        'Features_Used': selected_features})
        fetched_records = cursor.fetchall()

        conn.commit()
        conn.close()

        # Clear the treeview table
        self.training_model_tree.delete(*self.training_model_tree.get_children())

        # Add new data to the screen
        count = 0
        for record in fetched_records:
            self.training_model_tree.insert(parent='', index='end', iid=count, text='', values=(record[0],
                                                                                                record[1], record[2],
                                                                                                record[3], record[4],
                                                                                                record[5], record[6],
                                                                                                record[7], record[8],
                                                                                                record[9], record[10]))
            # Increment counter
            count += 1

    def on_column_clicked(self, event):
        region_clicked = self.training_model_tree.identify_region(event.x, event.y)

        if region_clicked not in 'heading':
            return
        if self.sorted_state == 'off':
            column_clicked = self.training_model_tree.identify_column(event.x)
            column_clicked_index = int(column_clicked[1:]) - 1

            self.sorted_state = 'on'
            column_clicked_name = (self.training_model_tree['columns'][column_clicked_index])

            # Puts a down arrow in the column name
            self.training_model_tree.heading(column_clicked_name, text=column_clicked_name + ' ' * 3 + 'V')

            conn = sqlite3.connect('Training_Model_Database')  # creates a database file and puts it in the directory

            # creates a cursor that does all the editing
            cursor = conn.cursor()

            # query the database (which means save) addresses=the table, oid = unique original id number
            cursor.execute('SELECT *, oid FROM training_model_table ORDER BY ' + column_clicked_name + ' DESC')
            fetched_records = cursor.fetchall()  # fetchone() brings back one record, fetchmany() brings many, fetchall() brings all records

            # Commit changes
            conn.commit()
            # Close connection
            conn.close()

            # Clear the treeview table
            self.training_model_tree.delete(*self.training_model_tree.get_children())

            # Refill the treeview table
            count = 0
            for record in fetched_records:
                self.training_model_tree.insert(parent='', index='end', iid=count, text='',
                                                values=(
                                                    record[0], record[1], record[2], record[3], record[4], record[5],
                                                    record[6], record[7],
                                                    record[8], record[9], record[10]))
                # Increment counter
                count += 1

        else:
            # Reload the original treeview data
            for column in self.training_model_tree['columns']:
                self.training_model_tree.heading(column, text=column)  # Reload the original treeview data

            self.sorted_state = 'off'

    def on_delete_selected_model(self):
        selected = self.training_model_tree.selection()[0]
        tree_values = self.training_model_tree.item(selected, 'values')

        # Create a database or connect to one that exists
        conn = sqlite3.connect('Training_Model_Database')

        # Create a cursor instance
        cursor = conn.cursor()

        # Delete from database
        cursor.execute(
            'DELETE from training_model_table WHERE Saved_Model_Name = :Saved_Model_Name AND Date_Created = :Date_Created AND Total_Average_Score = :Total_Average_Score',
            {'Date_Created': tree_values[0],
             'Dataframe': tree_values[2],
             'Saved_Model_Name': tree_values[3],
             'Total_Average_Score': tree_values[4]})
        # Commit changes
        conn.commit()

        # Close our connection
        conn.close()

        # Delete row from treeview
        self.training_model_tree.delete(selected)

        # Remove selection model from pickled data folder
        if os.path.exists(
                'C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/machine_learning_app/predictor/saved_training_pickle_models/' +
                tree_values[3] + '.pickle'):
            os.remove(
                'C:/Users/micha/Pycharm(Local)/LinearRegressionRepo/machine_learning_app/predictor/saved_training_pickle_models/' +
                tree_values[3] + '.pickle')

        # Add a removal message alert
        tree_removal_label = Label(self.training_model_button_frame, text='Selected Row Deleted', fg='red')
        tree_removal_label.grid(row=4, column=0, pady=5)

    def on_further_train(self):
        selected_treeview_row = self.training_model_tree.selection()[0]
        selected_model_name = self.training_model_tree.item(selected_treeview_row, 'values')[3]
        Trainer.existing_model_progress_tracker(self, self.selected_features,
                                                self.scaled_df, self.target_variable, selected_model_name,
                                                selected_treeview_row)

    def on_move_record_down(self):
        rows = self.training_model_tree.selection()
        for row in reversed(rows):
            self.training_model_tree.move(row, self.training_model_tree.parent(row),
                                          self.training_model_tree.index(row) + 1)

    def on_move_record_up(self):
        rows = self.training_model_tree.selection()
        for row in rows:
            self.training_model_tree.move(row, self.training_model_tree.parent(row),
                                          self.training_model_tree.index(row) - 1)

    def on_use_selected_model(self, e=None):
        selected = self.training_model_tree.selection()
        tree_values = self.training_model_tree.item(selected, 'values')

        from Step_10_Predicting.predictor import PredictorTreeviewPage
        PredictorTreeviewPage.selected_training_model = tree_values[3]
        print('Selected Training Model:', tree_values[3])

        training_model_window.destroy()

    def query_database(self):
        if self.filter_current_features_checkbox.instate(['selected']) == True:
            self.filter_features()
            return
        else:
            pass

        # creates a database if one doesn't already exist, otherwise it connects to it
        conn = sqlite3.connect('Training_Model_Database')  # creates a database file and puts it in the directory

        # creates a cursor that does all the editing
        cursor = conn.cursor()

        # Create table in database if it doesn't already exist
        cursor.execute("""CREATE TABLE if not exists training_model_table (
                Date_Created DATE,
                Target text,
                Dataframe text,
                Saved_Model_Name text,
                Total_Average_Score real,
                Total_Model_Upgrades integer,
                Last_Upgrade_Total_Runtimes integer,
                Total_Runtimes integer,
                Best_Average_Score real,
                Best_Score_Runtimes integer,
                Features_Used text
                )""")

        # query the database (which means save) addresses=the table, oid = unique original id number
        cursor.execute('SELECT *, oid FROM training_model_table')
        fetched_records = cursor.fetchall()  # fetchone() brings back one record, fetchmany() brings many, fetchall() brings all records

        # Commit changes
        conn.commit()
        # Close connection
        conn.close()

        # Clear the treeview table
        self.training_model_tree.delete(*self.training_model_tree.get_children())

        # Add our data to the screen
        global count
        count = 0
        for record in fetched_records:
            self.training_model_tree.insert(parent='', index='end', iid=count, text='', values=(record[0], record[1],
                                                                                                record[2], record[3],
                                                                                                record[4], record[5],
                                                                                                record[6], record[7],
                                                                                                record[8], record[9],
                                                                                                record[10]))
            # Increment counter
            count += 1

    def on_save_current_row_order(self):
        # Connect to database
        conn = sqlite3.connect('Training_Model_Database')

        # Create cursor
        cursor = conn.cursor()

        # Delete old data order
        cursor.execute('DELETE FROM training_model_table')

        # Add new record
        for record in self.training_model_tree.get_children():
            # Insert new reordered data into table
            cursor.execute(
                "INSERT INTO training_model_table VALUES (:Date_Created, :Target, :Dataframe, :Saved_Model_Name, :Total_Average_Score, :Total_Model_Upgrades, :Total_Runtimes, :Best_Average_Score, :Best_Score_Runtimes, :Features_Used)",
                {'Date_Created': self.training_model_tree.item(record[0], 'values')[0],
                 'Target': self.training_model_tree.item(record[0], 'values')[1],
                 'Dataframe': self.training_model_tree.item(record[0], 'values')[2],
                 'Saved_Model_Name': self.training_model_tree.item(record[0], 'values')[3],
                 'Total_Average_Score': self.training_model_tree.item(record[0], 'values')[4],
                 'Total_Model_Upgrades': self.training_model_tree.item(record[0], 'values')[5],
                 'Last_Upgrade_Total_Runtimes': self.training_model_tree.item(record[0], 'values')[6],
                 'Total_Runtimes': self.training_model_tree.item(record[0], 'values')[7],
                 'Best_Average_Score': self.training_model_tree.item(record[0], 'values')[8],
                 'Best_Score_Runtimes': self.training_model_tree.item(record[0], 'values')[9],
                 'Features_Used': self.training_model_tree.item(record[0], 'values')[10]
                 })

        conn.commit()
        conn.close()

        # Clear the treeview table
        self.training_model_tree.delete(*self.training_model_tree.get_children())

        # Refresh the database
        self.query_database()


class Trainer():
    def entry_box_initial_clearer(self, e, entry_box):
        entry_box.delete(0, END)

    def existing_model_progress_tracker(self, selected_feature_combination, scaled_dataframe, target_variable,
                                        selected_model_name, selected_treeview_row):
        self.state = 'existing'
        self.selected_treeview_row = selected_treeview_row

        # Makes sure only one window opens
        global further_training_progress_window
        try:
            if further_training_progress_window.state() == 'normal': further_training_progress_window.focus()
        except:
            # Create window
            further_training_progress_window = Toplevel(training_model_window)
            further_training_progress_window.title('Further Train Selected Model')
            further_training_progress_window.geometry('400x250')

            # Widgets
            spinbox_trainer_variable = StringVar(further_training_progress_window)
            self.training_runtimes_spinbox = Spinbox(further_training_progress_window, from_=500, to=100000000,
                                                     increment=500, textvariable=spinbox_trainer_variable,
                                                     font=('Ariel', 11, 'bold'), width=19)
            spinbox_trainer_variable.set('Runtimes')

            run_further_trainer_button = ttk.Button(further_training_progress_window, text='Run',
                                                    command=lambda: Trainer.main_trainer(self,
                                                                                         selected_feature_combination,
                                                                                         scaled_dataframe,
                                                                                         target_variable,
                                                                                         further_training_progress_window,
                                                                                         self.training_runtimes_spinbox,
                                                                                         selected_model_name,
                                                                                         selected_treeview_row))
            self.predicted_time_label = ttk.Label(further_training_progress_window, text='')
            global training_progress_bar
            training_progress_bar = ttk.Progressbar(further_training_progress_window, orient=HORIZONTAL, length=200,
                                                    mode='determinate')
            self.training_progress_label = ttk.Label(further_training_progress_window, text='')
            global model_upgrades_label
            self.model_upgrades_label = Label(further_training_progress_window, text='', fg='green')

            # Locations
            self.training_runtimes_spinbox.grid(row=0, column=0, padx=90)
            run_further_trainer_button.grid(row=1, column=0, padx=90, pady=5)
            self.predicted_time_label.grid(row=2, column=0, padx=90, pady=5)
            training_progress_bar.grid(row=4, column=0, padx=90, pady=5)
            self.training_progress_label.grid(row=5, column=0, padx=90, pady=5)
            self.model_upgrades_label.grid(row=6, column=0, padx=90)

            further_training_progress_window.mainloop()

    def main_trainer(self, selected_feature_combination, scaled_dataframe, target_variable, training_progress_window,
                     runtime_entry, saved_model_name, selected_treeview_row=None):
        self.saved_model_name = saved_model_name

        # Gives an error if the model name already matches a model name in the database
        # Connect to database
        conn = sqlite3.connect('Training_Model_Database')

        # Create cursor
        cursor = conn.cursor()

        if self.state == 'new':
            # Makes sure that no duplicates of model names are made
            cursor.execute('SELECT * FROM training_model_table WHERE Saved_Model_Name = :Saved_Model_Name',
                           {'Saved_Model_Name': self.saved_model_name})
            trainer_fetched_records = cursor.fetchall()
            if len(trainer_fetched_records) > 0:
                messagebox.showerror('Error',
                                     'Error: There is already a model named "' + self.saved_model_name + '" in the database.'
                                                                                                         '\nPlease insert a different name or delete the old model.')
                return
            conn.commit()
            conn.close()

        trainer_runtimes = int(runtime_entry.get())
        trainer_predicted_time_to_run = Trainer.trainer_time_predictor(self, selected_feature_combination,
                                                                       trainer_runtimes)

        training_progress_bar['value'] = 0
        self.predicted_time_label.config(text=('Predicted Time: ' + trainer_predicted_time_to_run))

        # Popup box
        trainer_response = messagebox.askyesno('Caution', 'Trainer will take:\n\n'
                                               + str(
            trainer_predicted_time_to_run) + '\n\n to complete, are you sure you want to continue?')
        training_progress_window.lift()
        training_model_window.lift()
        training_progress_window.lift()

        if trainer_response == True:
            pass
        else:
            return

        small_loops = 10
        start_time = time.time()

        selected_feature_combination.append(target_variable)

        df = scaled_dataframe[selected_feature_combination]

        X = np.array(df.drop([target_variable], axis=1))
        y = np.array(df[target_variable])

        save_pickle_to = 'saved_training_pickle_models/' + saved_model_name + '.pickle'

        current_model_regression_line = linear_model.LinearRegression()
        # split_object = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

        self.upgrades_to_pickle_model = 0
        for lap in range(int(trainer_runtimes)):
            training_progress_bar['value'] += (1 / int(trainer_runtimes) * 100)
            self.training_progress_label.config(text=str(format(training_progress_bar['value'], '.2f')) + '%')
            training_progress_window.update_idletasks()

            current_model_total_score, old_pickled_model_total_score = 0, 0
            for _ in range(small_loops):
                # ToDo figure out how to do stratified shuffle split
                # # Splits the data set
                # for train_index, test_index in split_object.split(df, df[target_variable]):
                #     stratified_training_set = df.loc[train_index]
                #     stratified_testing_set = df.loc[test_index]
                # X_train = stratified_training_set.drop([target_variable])
                # y_train = stratified_training_set[target_variable]
                # X_test = stratified_testing_set.drop([target_variable])
                # y_test = stratified_testing_set[target_variable]


                # The line below will only be for categorical testing
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

                current_model_regression_line.fit(X_train, y_train)
                current_model_score = current_model_regression_line.score(X_test, y_test)
                current_model_total_score = current_model_total_score + current_model_score

                if os.path.exists(save_pickle_to):
                    pickle_in = open(save_pickle_to, 'r+b')
                    old_pickled_regression_line = pickle.load(pickle_in)
                    old_pickled_model_score = old_pickled_regression_line.score(X_test, y_test)
                    old_pickled_model_total_score += old_pickled_model_score
                else:
                    old_pickled_model_score = -1000000
                    old_pickled_model_total_score += old_pickled_model_score

            global current_model_average_score
            current_model_average_score = current_model_total_score / small_loops
            old_pickled_model_average_score = old_pickled_model_total_score / small_loops

            if current_model_average_score > old_pickled_model_average_score:
                print('\033[32m' + '\n=======================Model Updated=======================')
                print('Runtime:', lap)
                # resets the coloring
                print('\033[39m')

                old_pickled_model_average_score = current_model_average_score
                self.best_average_score = current_model_average_score
                self.best_score_runtimes = trainer_runtimes
                self.upgrades_to_pickle_model = self.upgrades_to_pickle_model + 1
                self.exact_upgrade_runtimes = lap
                with open(save_pickle_to, 'wb') as f:
                    pickle.dump(current_model_regression_line, f)

            else:
                self.best_average_score = old_pickled_model_average_score
                if self.state == 'existing':
                    selected_row = self.selected_treeview_row
                    self.best_score_runtimes = self.training_model_tree.item(selected_row, 'values')[6]

        try:
            self.model_upgrades_label.config(text='Upgrades to Selected Model: ' + str(self.upgrades_to_pickle_model))
        except:
            pass

        # ToDo make it to where only one of each of the smaller windows can open

        if self.state == 'new':
            self.old_pickled_model_average_score = old_pickled_model_average_score
            self.trainer_runtimes = trainer_runtimes
            Trainer.training_model_database_inserter(self)
            self.best_score_runtimes = trainer_runtimes
            new_training_progress_window.destroy()
        else:
            # Update the old database numbers with the new numbers
            Trainer.training_database_updater(self, trainer_runtimes, current_model_average_score)

    def new_model_creator(self, selected_feature_combination, scaled_dataframe, target_variable):
        self.state = 'new'

        # Makes sure only one window will open at a time
        global new_training_progress_window
        try:
            if new_training_progress_window.state() == 'normal': new_training_progress_window.focus()
        except:
            # Create window
            new_training_progress_window = Toplevel(training_model_window)
            new_training_progress_window.title('Training Progress')
            new_training_progress_window.geometry('350x170')

            # widgets
            global training_model_name_entry
            training_model_name_entry = Entry(new_training_progress_window, font=('Ariel', 11, 'bold'))
            training_model_name_entry.insert(0, 'Model Name')
            training_model_name_entry.bind('<ButtonRelease-1>', lambda event,
                                                                       entry_box=training_model_name_entry: Trainer.entry_box_initial_clearer(
                self, event, training_model_name_entry))

            spinbox_trainer_variable = StringVar(new_training_progress_window)
            self.training_runtimes_spinbox = Spinbox(new_training_progress_window, from_=500, to=100000000,
                                                     increment=500, textvariable=spinbox_trainer_variable,
                                                     font=('Ariel', 11, 'bold'), width=19)
            spinbox_trainer_variable.set('Runtimes')
            create_new_model_run_button = ttk.Button(new_training_progress_window, text='Run',
                                                     command=lambda: Trainer.main_trainer(self,
                                                                                          selected_feature_combination,
                                                                                          scaled_dataframe,
                                                                                          target_variable,
                                                                                          new_training_progress_window,
                                                                                          self.training_runtimes_spinbox,
                                                                                          training_model_name_entry.get()))
            global training_progress_bar
            training_progress_bar = ttk.Progressbar(new_training_progress_window, orient=HORIZONTAL, length=200,
                                                    mode='determinate')
            self.training_progress_label = ttk.Label(new_training_progress_window, text='')
            self.predicted_time_label = ttk.Label(new_training_progress_window, text='')

            # Locations
            training_model_name_entry.grid(row=0, column=0, padx=40)
            self.training_runtimes_spinbox.grid(row=1, column=0, padx=40)
            create_new_model_run_button.grid(row=2, column=0, padx=40, pady=2)
            self.predicted_time_label.grid(row=3, column=0, padx=40, pady=5)
            training_progress_bar.grid(row=4, column=0, padx=40)
            self.training_progress_label.grid(row=5, column=0, padx=40)

            new_training_progress_window.mainloop()

    def trainer_time_predictor(self, selected_feature_combination, trainer_runtimes):
        # ToDo fix the time predictor
        trainer_predicted_time = .58439946 ** len(selected_feature_combination) * trainer_runtimes
        return functions.time_formatter(format(trainer_predicted_time, '.2f'))

    # ToDo make, upon first creation, the last upgrade total runtimes the exact runtime

    def training_database_updater(self, trainer_runtimes, current_model_average_score):
        selected_row = self.selected_treeview_row
        selected_tree_values = self.training_model_tree.item(selected_row, 'values')

        current_tree_total_average_score = float(selected_tree_values[4])
        current_tree_last_upgrade_total_runtimes = selected_tree_values[6]
        current_tree_total_runtimes = selected_tree_values[7]

        # Makes the calculations to keep the total values and averages accurate
        total_runtimes = int(current_tree_total_runtimes) + int(trainer_runtimes)
        weighted_ratio = int(trainer_runtimes) / total_runtimes
        print('current model average score:', current_model_average_score)
        print('weighted ratio:', weighted_ratio)
        print('current tree average score:', current_tree_total_average_score)
        print('current model weighted average:', current_model_average_score * weighted_ratio)
        print('tree old total average score:', float(current_tree_total_average_score) * (1 - weighted_ratio))
        total_average = ((current_model_average_score * weighted_ratio) + (
                float(current_tree_total_average_score) * (1 - weighted_ratio)))
        print('total average:', total_average)

        # ToDo delete upgrades to selected model after the run button is hit

        # Convert best average score to a comparable format
        self.best_average_score = float(str(self.best_average_score.tolist()))

        if current_tree_total_average_score > self.best_average_score:
            updater_best_average_score = current_tree_total_average_score
        else:
            updater_best_average_score = self.best_average_score

        # Updates the total model upgrades by summing them
        current_tree_upgrades = int(selected_tree_values[5])
        total_model_upgrades = self.upgrades_to_pickle_model + current_tree_upgrades

        if self.upgrades_to_pickle_model > 0:
            last_upgrade_total_runtimes = self.exact_upgrade_runtimes + int(current_tree_total_runtimes)
        else:
            last_upgrade_total_runtimes = current_tree_last_upgrade_total_runtimes

        # Connect to database
        conn = sqlite3.connect('Training_Model_Database')

        # Create cursor
        cursor = conn.cursor()

        # Update record
        cursor.execute("""UPDATE training_model_table SET
                       Total_Average_Score = :Total_Average_Score,
                       Total_Model_Upgrades = :Total_Model_Upgrades,
                       Last_Upgrade_Total_Runtimes = :Last_Upgrade_Total_Runtimes,
                       Total_Runtimes = :Total_Runtimes,
                       Best_Average_Score = :Best_Average_Score,
                       Best_Score_Runtimes = :Best_Score_Runtimes
                       WHERE Total_Average_Score = :Old_Total_Average_Score""",
                       {
                           'Total_Average_Score': round(total_average, 13),
                           'Total_Model_Upgrades': total_model_upgrades,
                           'Last_Upgrade_Total_Runtimes': last_upgrade_total_runtimes,
                           'Total_Runtimes': total_runtimes,
                           'Best_Average_Score': updater_best_average_score,
                           'Best_Score_Runtimes': self.best_score_runtimes,
                           'Old_Total_Average_Score': current_tree_total_average_score
                       })

        # Commit changes
        conn.commit()

        # Close connection
        conn.close()

        # Clear the treeview table
        self.training_model_tree.delete(*self.training_model_tree.get_children())

        # Reset tree by querying the database again
        self.query_database()

    def training_model_database_inserter(self):
        # Check to see if any upgrades were made
        try:
            exact_upgrade_runtimes = self.exact_upgrade_runtimes
        except:
            exact_upgrade_runtimes = 0

        selected_features = self.selected_features
        selected_features.remove(self.target_variable)
        selected_features = ', '.join(selected_features)

        # Connect to database
        conn = sqlite3.connect('Training_Model_Database')

        # Create cursor
        cursor = conn.cursor()

        # Add new record
        cursor.execute(
            "INSERT INTO training_model_table VALUES (:Date_Created, :Target, :Dataframe, :Saved_Model_Name, :Total_Average_Score, :Total_Model_Upgrades, :Last_Upgrade_Total_Runtimes, :Total_Runtimes, :Best_Average_Score, :Best_Score_Runtimes, :Features_Used)",
            {
                'Date_Created': datetime.date.today(),
                'Target': self.target_variable,
                'Dataframe': str(self.csv_name),
                'Saved_Model_Name': self.saved_model_name,
                'Total_Average_Score': round(self.old_pickled_model_average_score, 13),
                'Total_Model_Upgrades': self.upgrades_to_pickle_model,
                'Last_Upgrade_Total_Runtimes': exact_upgrade_runtimes,
                'Total_Runtimes': self.trainer_runtimes,
                'Best_Average_Score': round(self.old_pickled_model_average_score, 13),
                'Best_Score_Runtimes': self.trainer_runtimes,
                'Features_Used': selected_features
            })

        # Commit changes
        conn.commit()

        # Close connection
        conn.close()

        # Clear the treeview table
        self.training_model_tree.delete(*self.training_model_tree.get_children())

        # Reset tree by querying the database again
        self.query_database()


if __name__ == '__main__':
    pass

    # selected_feature_combination = feature_grabber()
    # print('Most Important Features:', selected_feature_combination)
    # print('Length of Features:', len(selected_feature_combination))
    #
    # microprocessors = 5
    # predicted_time = .7199564681 ** len(selected_feature_combination) * processor_runs * microprocessors
    # # ToDo fix time predictor
    # print('Predicted Run Time:', functions.time_formatter(predicted_time))
    # user_input = input('Run Trainer? Hit ENTER for yes: ')
    # if user_input == '':
    #     pass
    # else:
    #     quit()
    #
    #
    # trainer_runtimes = 100
    # start_time = time.time()
    #
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     # this is how you get a return from multiprocessing
    #     f1 = executor.submit(main_trainer, int(processor_runs/microprocessors), selected_feature_combination, predicted_time)
    #     f2 = executor.submit(main_trainer, int(processor_runs/microprocessors), selected_feature_combination, predicted_time)
    #     f3 = executor.submit(main_trainer, int(processor_runs/microprocessors), selected_feature_combination, predicted_time)
    #     f4 = executor.submit(main_trainer, int(processor_runs/microprocessors), selected_feature_combination, predicted_time)
    #     f5 = executor.submit(main_trainer, int(processor_runs/microprocessors), selected_feature_combination, predicted_time)
    #
    #
    # total_pickle_model_upgrades = f1.result() + f2.result() + f3.result() + f4.result() + f5.result()
    # elapsed_time = time.time() - start_time
    # # ToDo make this doable with multiprocessors again
    # print('\n\nTotal Pickle Model upgrades:', total_pickle_model_upgrades)
    # print('Elapsed Time:', functions.time_formatter(elapsed_time))
    # print('Predicted Run Time:', functions.time_formatter(predicted_time))
    #
    #
    # if elapsed_time > 30:
    #     functions.email_or_text_alert('Trainer:', 'Total Pickle Model upgrades:' + str(total_pickle_model_upgrades),
    #     '4052198820@mms.att.net')
    #     quit()
    # else:
    #     quit()
