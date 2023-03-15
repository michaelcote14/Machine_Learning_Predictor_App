import datetime
import sqlite3
import threading
from tkinter import *
from tkinter import ttk

import itertools
import numpy as np
import rfpimp
import sklearn
import time
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from Extras import functions
from Extras.functions import time_formatter


class MostImportantFeaturesPage:
    def __init__(self, target_variable, scaled_df, csv_name, image_changer, green_check_label_features):
        self.target_variable = target_variable
        self.scaled_df = scaled_df
        self.csv_name = csv_name
        self.image_changer = image_changer
        self.green_check_label_features = green_check_label_features
        # Used for sorting the dataframe
        self.sorted_state = 'off'

        # Makes sure only one window will open at a time
        global feature_selection_window
        try:
            if feature_selection_window.state() == 'normal': feature_selection_window.focus()
        except:
            # Create window
            feature_selection_window = Toplevel(bg='gray10')
            feature_selection_window.title('Feature Selection')
            feature_selection_window.geometry('1300x350')

            # Create a frame for tree
            feature_selection_tree_frame = Frame(feature_selection_window, bg='gray10')
            feature_selection_tree_frame.pack(ipadx=250)
            # Create a frame for buttons
            self.feature_selection_button_frame = Frame(feature_selection_window, bg='gray10')
            self.feature_selection_button_frame.pack(ipadx=300, fill='x', expand=True)

            # Create tree
            self.feature_selection_tree = ttk.Treeview(feature_selection_tree_frame)

            # Create scrollbar for the tree frame
            self.feature_selection_horizontal_scrollbar = ttk.Scrollbar(feature_selection_tree_frame, orient=HORIZONTAL,
                                                                        command=self.feature_selection_tree.xview)
            self.feature_selection_horizontal_scrollbar.pack(side=BOTTOM, fill=X)
            self.feature_selection_tree.configure(xscrollcommand=self.feature_selection_horizontal_scrollbar.set)

            # Define columns
            self.feature_selection_tree['columns'] = (
                'Date_Created', 'Target', 'Dataframe', 'Most_Important_Features')

            # Format columns
            self.feature_selection_tree.column('#0', width=0, stretch=NO)
            self.feature_selection_tree.column('Date_Created', anchor=W, width=120, stretch=NO)
            self.feature_selection_tree.column('Target', anchor=W, width=110, stretch=NO)
            self.feature_selection_tree.column('Dataframe', anchor=W, width=210, stretch=NO)
            self.feature_selection_tree.column('Most_Important_Features', anchor=CENTER, minwidth=850)

            # Create headings
            self.feature_selection_tree.heading('Date_Created', text='Date_Created', anchor=W)
            self.feature_selection_tree.heading('Target', text='Target', anchor=W)
            self.feature_selection_tree.heading('Dataframe', text='Dataframe', anchor=W)
            self.feature_selection_tree.heading('Most_Important_Features', text='Most_Important_Features', anchor=CENTER)

            # Bind treeview
            self.feature_selection_tree.bind('<Double-Button-1>', self.use_selected_features)
            self.feature_selection_tree.bind('<Button-1>', self.on_column_clicked)

            # Widgets
            use_selected_features_button = ttk.Button(self.feature_selection_button_frame, text='Use Selected Features',
                                                      command=self.use_selected_features)
            create_new_features_button = ttk.Button(self.feature_selection_button_frame,
                                                    text='Create New Feature Combination',
                                                    command=self.create_new_feature_combination)
            delete_selected_button = ttk.Button(self.feature_selection_button_frame, text='Delete Selected Row',
                                                command=self.remove_selected_row)
            up_button = ttk.Button(self.feature_selection_button_frame, text='Move Selection Up',
                                   command=self.move_record_up)
            down_button = ttk.Button(self.feature_selection_button_frame, text='Move Selection Down',
                                     command=self.move_record_down)
            save_row_order_button = ttk.Button(self.feature_selection_button_frame, text='Save Current Row Order',
                                               command=feature_selection_window)
            self.filter_current_dataframe_checkbox = ttk.Checkbutton(self.feature_selection_button_frame,
                                                                     text='Only Show Current Target and Dataframe',
                                                                     command=self.filter_dataframe)

            # Makes the current dataframe checkbox start out as selected
            self.filter_current_dataframe_checkbox.state(['!alternate'])
            self.filter_current_dataframe_checkbox.state(['selected'])

            # Locations
            self.feature_selection_tree.pack(expand=True, fill=BOTH)

            create_new_features_button.grid(row=1, column=0, padx=20, pady=5)
            use_selected_features_button.grid(row=2, column=0, padx=20, pady=5)

            up_button.grid(row=1, column=1, padx=20, pady=5)
            down_button.grid(row=2, column=1, padx=20, pady=5)
            save_row_order_button.grid(row=3, column=1, padx=20, pady=5)

            delete_selected_button.grid(row=3, column=0, padx=20, pady=5)
            self.filter_current_dataframe_checkbox.grid(row=1, column=2, padx=20, pady=5)

            self.query_database()

    def create_new_feature_combination(self):
        def indeterminate_progress_threader():
            self.feature_selection_progressbar = ttk.Progressbar(feature_selection_progress_window, orient=HORIZONTAL,
                                                                 length=300, mode='indeterminate')
            self.feature_selection_progressbar.grid(row=0, column=0, columnspan=3, padx=10)

            self.feature_selection_progressbar.start(8)

            ImportantFeaturesFinder.feature_importer(self, self.feature_selection_runtimes_spinbox.get(),
                                                     self.scaled_df, self.target_variable,
                                                     self.amount_of_features_combo.get(), 'yes')

        def important_features_creator():
            importance_creator_predicted_time = ImportantFeaturesFinder.importance_time_predictor(self,
                                                                                                  self.feature_selection_runtimes_spinbox.get(),
                                                                                                  self.amount_of_features_combo.get(),
                                                                                                  self.scaled_df)
            # print('Importance Creator Predicted Time:', importance_creator_predicted_time)

            # ToDo fix time predictor for important feature creator and feature combiner

            # Widgets
            self.importance_finder_caution_label = Label(feature_selection_progress_window,
                                                         text='Step 1 will take approximately\n'
                                                              + importance_creator_predicted_time + '\nto run, are you sure you wan to continue?')

            self.importance_finder_yes_button = ttk.Button(feature_selection_progress_window, text='Yes',
                                                           command=threading.Thread(
                                                               target=indeterminate_progress_threader).start)

            self.importance_finder_no_button = ttk.Button(feature_selection_progress_window, text='No',
                                                          command=lambda: ImportantFeaturesFinder.feature_importer(self,
                                                                                                                   self.feature_selection_runtimes_spinbow.get(),
                                                                                                                   self.scaled_df,
                                                                                                                   self.target_variable,
                                                                                                                   self.amount_of_features_combo.get(),
                                                                                                                   'no'))

            # Locations
            self.importance_finder_caution_label.grid(row=6, column=1)
            self.importance_finder_yes_button.grid(row=7, column=1, padx=(0, 80))
            self.importance_finder_no_button.grid(row=7, column=1, padx=(80, 0))

        def feature_selection_progress_tracker():
            global feature_selection_progress_window
            try:
                if feature_selection_progress_window.state() == 'normal': feature_selection_progress_window.focus()
            except:
                feature_selection_progress_window = Toplevel(feature_selection_window)
                feature_selection_progress_window.title('Feature Selection Progress')
                feature_selection_progress_window.geometry('328x220')

                # Widgets
                spinbox_variable = StringVar(feature_selection_progress_window)
                self.feature_selection_runtimes_spinbox = Spinbox(feature_selection_progress_window, from_=500,
                                                                  to=10000000,
                                                                  increment=500, textvariable=spinbox_variable,
                                                                  font=('Ariel', 11), width=18)
                spinbox_variable.set('Runtimes')

                # Amount of features for combo box
                amount_of_features_list = [x for x in range(len(self.scaled_df.columns))]
                amount_of_features_list.insert(0, 'Select # of Features...')
                self.amount_of_features_combo = ttk.Combobox(feature_selection_progress_window,
                                                             values=amount_of_features_list, width=23)
                self.amount_of_features_combo.current(0)
                create_new_features_run_button = ttk.Button(feature_selection_progress_window, text='Run',
                                                            command=important_features_creator)
                self.feature_selection_progress_label = Label(feature_selection_progress_window, text='')
                self.feature_selection_predicted_time_label = Label(feature_selection_progress_window, text='')

                # Locations
                self.feature_selection_progress_label.grid(row=1, column=1, padx=80)
                self.feature_selection_predicted_time_label.grid(row=2, column=1, padx=80)
                self.feature_selection_runtimes_spinbox.grid(row=3, column=1, padx=80)
                self.amount_of_features_combo.grid(row=4, column=1, padx=80)
                create_new_features_run_button.grid(row=5, column=1, padx=80)

                feature_selection_progress_window.mainloop()

        feature_selection_progress_tracker()

    def current_row_saver(self):
        # Connect to database
        conn = sqlite3.connect('../../Databases/Most_Important_Features_Database')

        # Create cursor
        cursor = conn.cursor()

        # Delete old data order
        cursor.execute('DELETE FROM most_important_features_table')

        # Add new record
        for record in self.feature_selection_tree.get_children():
            # Insert new reordered data into table
            cursor.execute(
                "INSERT INTO most_important_features_table VALUES (:Date_Created, :Dataframe, :Most_Important_Features)",
                {'Date_Created': self.feature_selection_tree.item(record[0], 'values')[0],
                 'Target': self.feature_selection_tree.item(record[0], 'values')[1],
                 'Dataframe': self.feature_selection_tree.item(record[0], 'values')[2],
                 'Most_Important_Features': self.feature_selection_tree.item(record[0], 'values')[3],
                 })

        # Commit changes
        conn.commit()
        # Close connection
        conn.close()

        # Clear the treeview table
        self.feature_selection_tree.delete(*self.feature_selection_tree.get_children())

        # Refresh the database
        self.query_database()

    def filter_dataframe(self):
        if self.filter_current_dataframe_checkbox.instate(['selected']) == True:
            pass
        else:
            # Clear the treeview table
            self.feature_selection_tree.delete(*self.feature_selection_tree.get_children())
            self.query_database()
            return

        # creates a database if one doesn't already exist, otherwise it connects to it
        conn = sqlite3.connect(
            '../../Databases/Most_Important_Features_Database')  # creates a database file and puts it in the directory

        # creates a cursor that does all the editing
        cursor = conn.cursor()

        # Grab only data you want
        cursor.execute('SELECT * FROM most_important_features_table WHERE Dataframe = :Dataframe AND Target = :Target',
                       {'Dataframe': self.csv_name,
                        'Target': self.target_variable})

        fetched_records = cursor.fetchall()

        conn.commit()
        conn.close()

        # Clear the treeview table
        self.feature_selection_tree.delete(*self.feature_selection_tree.get_children())

        # Add new data to the screen
        count = 0
        for record in fetched_records:
            self.feature_selection_tree.insert(parent='', index='end', iid=count, text='',
                                               values=(
                                                   record[0], record[1], record[2], record[3]))
            # Increment counter
            count += 1

    def move_record_down(self):
        rows = self.feature_selection_tree.selection()
        for row in reversed(rows):
            self.feature_selection_tree.move(row, self.feature_selection_tree.parent(row),
                                             self.feature_selection_tree.index(row))

    def move_record_up(self):
        rows = self.feature_selection_tree.selection()
        for row in rows:
            self.feature_selection_tree.move(row, self.feature_selection_tree.parent(row),
                                             self.feature_selection_tree.index(row) - 1)

    def on_column_clicked(self, event):
        region_clicked = self.feature_selection_tree.identify_region(event.x, event.y)

        if region_clicked not in 'heading':
            return
        if self.sorted_state == 'off':
            column_clicked = self.feature_selection_tree.identify_column(event.x)
            column_clicked_index = int(column_clicked[1:]) - 1

            self.sorted_state = 'on'
            column_clicked_name = (self.feature_selection_tree['columns'][column_clicked_index])
            print('column clicked name:', column_clicked_name)

            # Puts a down arrow in the column name
            self.feature_selection_tree.heading(column_clicked_name, text=column_clicked_name + ' ' * 3 + 'V')

            conn = sqlite3.connect(
                '../../Databases/Most_Important_Features_Database')  # creates a database file and puts it in the directory

            # creates a cursor that does all the editing
            cursor = conn.cursor()

            # query the database (which means save) addresses=the table, oid = unique original id number
            cursor.execute('SELECT *, oid FROM most_important_features_table ORDER BY ' + column_clicked_name + ' DESC')
            fetched_records = cursor.fetchall()  # fetchone() brings back one record, fetchmany() brings many, fetchall() brings all records

            # Commit changes
            conn.commit()
            # Close connection
            conn.close()

            # Clear the treeview table
            self.feature_selection_tree.delete(*self.feature_selection_tree.get_children())

            # Refill the treeview table
            count = 0
            for record in fetched_records:
                print('records in sorted function:', record)
                self.feature_selection_tree.insert(parent='', index='end', iid=count, text='',
                                                   values=(
                                                       record[0], record[1], record[2], record[3], record[4],
                                                       record[5]))
                # Increment counter
                count += 1


        else:
            # Reload the original treeview data
            for column in self.feature_selection_tree['columns']:
                self.feature_selection_tree.heading(column, text=column)  # Reload the original treeview data

            self.sorted_state = 'off'

    def query_database(self):
        # The below lines must be commented out to recreate the database
        if self.filter_current_dataframe_checkbox.instate(['selected']) == True:
            self.filter_dataframe()
            return
        else:
            pass

        # creates a database if one doesn't already exist, otherwise it connects to it
        conn = sqlite3.connect(
            '../../Databases/Most_Important_Features_Database')  # creates a database file and puts it in the directory

        # creates a cursor that does all the editing
        cursor = conn.cursor()

        # Create table in database if it doesn't already exist
        cursor.execute("""CREATE TABLE if not exists most_important_features_table (
                Date_Created DATE,
                Target text,
                Dataframe text,
                Most_Important_Features text)""")

        # query the database (which means save) addresses=the table, oid = unique original id number
        cursor.execute('SELECT *, oid FROM most_important_features_table')
        fetched_records = cursor.fetchall()  # fetchone() brings back one record, fetchmany() brings many, fetchall() brings all records

        # Commit changes
        conn.commit()
        # Close connection
        conn.close()

        # Clear the treeview
        self.feature_selection_tree.delete(*self.feature_selection_tree.get_children())

        # Add our data to the treeview screen
        global count
        count = 0
        for record in fetched_records:
            self.feature_selection_tree.insert(parent='', index='end', iid=count, text='',
                                               values=(
                                                   record[0], record[1], record[2], record[3]))
            # Increment counter
            count += 1

    def remove_selected_row(self):
        selected = self.feature_selection_tree.selection()[0]
        tree_values = self.feature_selection_tree.item(selected, 'values')
        # Create a database or connect to one that exists
        conn = sqlite3.connect('../../Databases/Most_Important_Features_Database')

        # Create a cursor instance
        cursor = conn.cursor()

        # Delete from database
        cursor.execute(
            'DELETE from most_important_features_table WHERE Date_Created = :Date_Created AND Most_Important_Features = :Most_Important_Features',
            {
                'Date_Created': tree_values[0],
                'Target': tree_values[1],
                'Dataframe': tree_values[2],
                'Most_Important_Features': tree_values[3],
            })

        # Commit changes
        conn.commit()

        # Close our connection
        conn.close()

        # Add a removal message alert
        tree_removal_label = Label(self.feature_selection_button_frame, text='Selected Row Deleted', fg='red')
        tree_removal_label.grid(row=2, column=2, pady=5, padx=5)

        # Remove selection from treeview
        self.feature_selection_tree.delete(selected)

    def use_selected_features(self, e=None):
        selected = self.feature_selection_tree.selection()[0]
        tree_values = self.feature_selection_tree.item(selected, 'values')

        selected_features = tree_values[3].split(', ')
        feature_selection_window.destroy()

        from Step_8_Training.trainer import TrainingModelPage
        TrainingModelPage.selected_features = selected_features

        from Step_10_Predicting.predictor import PredictorTreeviewPage
        PredictorTreeviewPage.selected_features = selected_features
        print('Selected Features:', selected_features)

        # Changes the red x to a green checkmark
        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_features, 24, 24)


class ImportantFeaturesFinder:
    def important_features_database_inserter(self, most_important_features):
        # Connect to database
        conn = sqlite3.connect('../../Databases/Most_Important_Features_Database')

        # Create cursor
        cursor = conn.cursor()

        # Add new record
        cursor.execute(
            "INSERT INTO most_important_features_table VALUES (:Date_Created, :Target, :Dataframe, :Most_Important_Features)",
            {
                'Date_Created': datetime.date.today(),
                'Dataframe': str(self.csv_name),
                'Target': self.target_variable,
                'Most_Important_Features': ', '.join(most_important_features)
            })

        # Commit changes
        conn.commit()

        # Close connection
        conn.close()

        # Clear the treeview table
        self.feature_selection_tree.delete(*self.feature_selection_tree.get_children())

        # Reset tree by querying the database again
        self.query_database()

        feature_selection_progress_window.destroy()
    def importance_time_predictor(self, runtimes, amount_of_features_selected, scaled_df):
        dataframe_column_length = scaled_df.shape[1]
        dataframe_row_length = scaled_df.shape[0]
        predicted_time = ((int(runtimes) * dataframe_column_length) + int(
            amount_of_features_selected) * 0.00016327 + dataframe_row_length * 0.00016327) * 0.00016327
        predicted_time = time_formatter(predicted_time)
        return predicted_time

    def feature_importer(self, runtimes, scaled_df, target_variable, amount_of_features, choice):
        start_time = time.time()
        if choice.lower() == 'yes':
            pass
        else:
            feature_selection_progress_window.destroy()
            return

        ######################################## Data preparation #########################################
        features = scaled_df.columns.tolist()
        ######################################## Train/test split #########################################

        scaled_df_train, scaled_df_test = train_test_split(scaled_df, test_size=0.20, random_state=0)
        scaled_df_train = scaled_df_train[features]
        scaled_df_test = scaled_df_test[features]

        X_train, y_train = scaled_df_train.drop(target_variable, axis=1), scaled_df_train[target_variable]
        X_test, y_test = scaled_df_test.drop(target_variable, axis=1), scaled_df_test[target_variable]

        # ################################################ Train #############################################
        #
        rf = RandomForestRegressor(n_estimators=int(runtimes), n_jobs=-1)
        rf.fit(X_train, y_train)
        #
        # ############################### Permutation feature importance #####################################
        #
        imp = rfpimp.importances(rf, X_test, y_test)

        # turn this into a dictionary
        importance_list = imp.index.tolist()
        importance_dictionary = {}
        loop_number = 0
        for i in importance_list:
            importance_dictionary[i] = imp['Importance'][loop_number]
            loop_number = loop_number + 1

        sorted_dict = dict(sorted(importance_dictionary.items(), key=lambda x: x[1], reverse=True))

        most_important_features = []
        for n in range(int(amount_of_features)):
            new_corr_list = list(sorted_dict)
            # print('new corr list:', new_corr_list)
            most_important_features.append(new_corr_list[n])
        most_important_values = list(sorted_dict.values())
        most_important_features.insert(0, target_variable)

        elapsed_time = time.time() - start_time

        self.feature_selection_progress_label.config(text='Completed')

        self.most_important_features = most_important_features

        ImportantFeaturesFinder.important_features_database_inserter(self, most_important_features)