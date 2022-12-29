import sklearn
from sklearn import linear_model
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle
import time
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast
from tkinter import *
from tkinter import ttk
import sqlite3
import datetime
from tkinter import messagebox
import mplcursors as mpc


class PredictorTreeviewPage:
    def __init__(self, scaled_df, target_variable, scaled_data_we_know_df, csv_name, data_we_know_dict):
        self.scaled_df = scaled_df
        self.target_variable = target_variable
        self.scaled_data_we_know_df = scaled_data_we_know_df
        self.csv_name = csv_name
        self.data_we_know_dict = data_we_know_dict

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
            predictor_window = Toplevel()
            predictor_window.title('Predictor')
            predictor_window.geometry('1100x350')
    
            # Create frame for treeview
            predictor_treeview_frame = Frame(predictor_window)
            predictor_treeview_frame.pack(ipadx=200)
            # Create frame for buttons
            self.predictor_buttons_frame = Frame(predictor_window)
            self.predictor_buttons_frame.pack(ipadx=200)
    
            # Create tree
            self.predictor_tree = ttk.Treeview(predictor_treeview_frame)
    
            # Create scrollbar for the tree
            predictor_horizontal_scrollbar = ttk.Scrollbar(predictor_treeview_frame, orient=HORIZONTAL, command=self.predictor_tree.xview)
            predictor_horizontal_scrollbar.pack(side=BOTTOM, fill=X)
            self.predictor_tree.configure(xscrollcommand=predictor_horizontal_scrollbar.set)
    
            # Define columns
            self.predictor_tree['columns'] = ('Date_Predicted', 'Target', 'Predicted_Value', 'Mean_Absolute_Error', 'Score', 'Cross_Val_Score', 'Data_Known', 'Model_Used', 'Dataframe', 'Features_Used', 'All_Pickle_Model_Predictions', 'Tested_Actual_Values')
    
            # Format columns
            self.predictor_tree.column('#0', width=0, stretch=NO)
            self.predictor_tree.column('Date_Predicted', anchor=W, width=110, stretch=NO)
            self.predictor_tree.column('Target', anchor=W, width=110, stretch=NO)
            self.predictor_tree.column('Predicted_Value', anchor=W, width=130, stretch=NO)
            self.predictor_tree.column('Mean_Absolute_Error', anchor=W, width=150, stretch=NO)
            self.predictor_tree.column('Score', anchor=W, width=120, stretch=NO)
            self.predictor_tree.column('Cross_Val_Score', anchor=W, width=120, stretch=NO)
            self.predictor_tree.column('Data_Known', anchor=W, width=200, stretch=NO)
            self.predictor_tree.column('Model_Used', anchor=W, width=140, stretch=NO)
            self.predictor_tree.column('Dataframe', anchor=W, width=140, stretch=NO)
            self.predictor_tree.column('Features_Used', anchor=W, width=450, stretch=NO)
            self.predictor_tree.column('All_Pickle_Model_Predictions', anchor=W, width=0, stretch=NO)
            self.predictor_tree.column('Tested_Actual_Values', anchor=W, width=0, stretch=NO)
            #ToDo put a graph comparer in or even a prediction comparer?
    
            # Create headings
            self.predictor_tree.heading('Date_Predicted', text='Date_Predicted', anchor=W)
            self.predictor_tree.heading('Target', text='Target', anchor=W)
            self.predictor_tree.heading('Predicted_Value', text='Predicted_Value', anchor=W)
            self.predictor_tree.heading('Mean_Absolute_Error', text='Mean_Absolute_Error', anchor=W)
            self.predictor_tree.heading('Score', text='Score', anchor=W)
            self.predictor_tree.heading('Cross_Val_Score', text='Cross_Val_Score', anchor=W)
            self.predictor_tree.heading('Data_Known', text='Data_Known', anchor=W)
            self.predictor_tree.heading('Model_Used', text='Model_Used', anchor=W)
            self.predictor_tree.heading('Dataframe', text='Dataframe', anchor=W)
            self.predictor_tree.heading('Features_Used', text='Features_Used', anchor=W)
            self.predictor_tree.heading('All_Pickle_Model_Predictions', text='All_Pickle_Model_Predictions', anchor=W)
            self.predictor_tree.heading('Tested_Actual_Values', text='Tested_Actual_Values', anchor=W)
    
            # Bind treeview to click for filter
            self.predictor_tree.bind('<Button-1>', self.on_column_clicked)
    
    
            # Widgets
            create_new_prediction_button = ttk.Button(self.predictor_buttons_frame, text='Create New Prediction', command=self.on_create_new_prediction)
            delete_selected_row_button = ttk.Button(self.predictor_buttons_frame, text='Delete Selected Row', command=self.on_delete_selected_row)
            up_button = ttk.Button(self.predictor_buttons_frame, text='Move Record Up', command=self.on_move_record_up)
            down_button = ttk.Button(self.predictor_buttons_frame, text='Move Record Down', command=self.on_move_record_down)
            row_order_saver_button = ttk.Button(self.predictor_buttons_frame, text='Save Current Row Order', command=self.on_save_current_row_order)
            self.filter_current_model_checkbox = ttk.Checkbutton(self.predictor_buttons_frame, text='Only Show Current Model', command=self.filter_model)
            graph_prediction_button = ttk.Button(self.predictor_buttons_frame, text='Graph Prediction', command=self.on_graph_prediction)
            prediction_analysis_button = ttk.Button(self.predictor_buttons_frame, text='Analyze Selected Prediction', command=self.on_prediction_analysis)
            test_against_file_button = ttk.Button(self.predictor_buttons_frame, text='Test Against File', command=on_test_against_file_button)
    
            # Makes the current dataframe checkbox start out as selected
            self.filter_current_model_checkbox.state(['!alternate'])
            self.filter_current_model_checkbox.state(['selected'])
    
            # Locations
            self.predictor_tree.pack(expand=True, fill=BOTH, padx=5)
            create_new_prediction_button.grid(row=0, column=0)
            delete_selected_row_button.grid(row=1, column=0)
            up_button.grid(row=0, column=1)
            down_button.grid(row=1, column=1)
            row_order_saver_button.grid(row=2, column=1)
            self.filter_current_model_checkbox.grid(row=0, column=2)
            graph_prediction_button.grid(row=2, column=0)
            prediction_analysis_button.grid(row=1, column=2)
            test_against_file_button.grid(row=2, column=2)
    
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
        conn = sqlite3.connect('Predictions_Database')  # creates a database file and puts it in the directory

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
                                                record[5], record[6], record[7], record[8], record[9], record[10], record[11]))
            # Increment counter
            count += 1


    def on_column_clicked(self, event):
        region_clicked = self.predictor_tree.identify_region(event.x, event.y)

        if region_clicked not in 'heading':
            return
        if self.sorted_state == 'off':
            column_clicked = self.predictor_tree.identify_column(event.x)
            column_clicked_index = int(column_clicked[1:])-1

            self.sorted_state = 'on'
            column_clicked_name = (self.predictor_tree['columns'][column_clicked_index])

            # Puts a down arrow in the column name
            self.predictor_tree.heading(column_clicked_name, text=column_clicked_name + ' ' * 3 + 'V')

            conn = sqlite3.connect('Predictions_Database')  # creates a database file and puts it in the directory

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
                        values=(record[0], record[1], record[2], record[3], record[4], record[5], record[6], record[7], record[8], record[9], record[10], record[11]))
                # Increment counter
                count += 1

        else:
            # Reload the original treeview data
            for column in self.predictor_tree['columns']:
                self.predictor_tree.heading(column, text=column) # Reload the original treeview data

            self.sorted_state = 'off'

    def on_create_new_prediction(self):
        def prediction_database_inserter():
            self.all_pickle_model_predictions, tested_actual_values = Predictor.predictor(self)

            # convert the above to usable sql format
            tested_actual_values = tested_actual_values.tolist()
            self.all_tree_pickle_model_predictions = ', '.join(map(str, self.all_pickle_model_predictions))
            tested_actual_values = ', '.join(map(str, tested_actual_values))


            # Connect to database
            conn = sqlite3.connect('Predictions_Database')

            # Create cursor
            cursor = conn.cursor()

            database_feature_combination = ', '.join(self.selected_features)

            # Makes the data we know dict nothing if a test dataframe was selected
            try:
                print(len(list(self.data_we_know_dict.values())[0]))
            except:
                self.data_we_know_dict = {}

            # Add new record
            cursor.execute("INSERT INTO predictions_table VALUES (:Date_Predicted, :Target, :Predicted_Value, :Mean_Absolute_Error, :Score, :Cross_Val_Score, :Data_Known, :Model_Used, :Dataframe, :Features_Used, :All_Pickle_Model_Predictions, :Tested_Actual_Values)",
                           {'Date_Predicted': datetime.date.today(),
                            'Target': self.target_variable,
                            'Predicted_Value': round(self.pickle_predicted_value, 15),
                            'Mean_Absolute_Error': self.pickle_model_average_mae,
                            'Score': round(self.pickle_model_average_score, 15),
                            'Cross_Val_Score': round(self.pickle_cross_val_average_score, 15),
                            'Data_Known': str(self.data_we_know_dict),
                            'Model_Used': self.selected_training_model,
                            'Dataframe': str(self.csv_name),
                            'Features_Used': database_feature_combination,
                            'All_Pickle_Model_Predictions': self.all_tree_pickle_model_predictions,
                            'Tested_Actual_Values': tested_actual_values})

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
        conn = sqlite3.connect('Predictions_Database')

        # Create a cursor instance
        cursor = conn.cursor()

        # Delete from database
        cursor.execute('DELETE from predictions_table WHERE Date_Predicted = :Date_Predicted AND Target = :Target AND Predicted_Value = :Predicted_Value AND Dataframe = :Dataframe AND Model_Used = :Model_Used AND Score = :Score',
                       {'Date_Predicted': tree_values[0],
                        'Target': tree_values[1],
                        'Predicted_Value': tree_values[2],
                        'Score': tree_values[4],
                        'Model_Used': tree_values[6],
                        'Dataframe': tree_values[7]})
        # Commit changes
        conn.commit()

        # Close our connection
        conn.close()

        # Clear the treeview table
        self.predictor_tree.delete(*self.predictor_tree.get_children())

        # Reset tree by querying the database again
        self.query_database()

        row_deleted_label = Label(self.predictor_buttons_frame, text='Selected Row Removed', fg='red')
        row_deleted_label.grid(row=1, column=2)

    def on_graph_prediction(self):
        # Get all predicted and test values from treeview (these are hidden and are not displayed in the treeview)
        selected = self.predictor_tree.selection()[0]
        print('selected:', selected)
        print('all items:', self.predictor_tree.item(selected, 'values'))
        tree_all_predicted_values = self.predictor_tree.item(selected, 'values')[10]
        print('tree all predicted values:', tree_all_predicted_values)
        tree_tested_actual_values = self.predictor_tree.item(selected, 'values')[11]
        print('tree tested actual values:', tree_tested_actual_values)

        # Split long string from treeview into a list
        tree_all_predicted_values = tree_all_predicted_values.split(', ')
        tree_tested_actual_values = tree_tested_actual_values.split(', ')


        # Turn tree values into arrays for plotter
        tree_all_predicted_values = np.array([float(value) for value in tree_all_predicted_values])
        tree_tested_actual_values = np.array([float(value) for value in tree_tested_actual_values])

        # Run plotter
        Predictor.predictor_plotter(self, tree_all_predicted_values, tree_tested_actual_values, self.target_variable)


    def on_move_record_down(self):
        rows = self.predictor_tree.selection()
        for row in reversed(rows):
            self.predictor_tree.move(row, self.predictor_tree.parent(row), self.predictor_tree.index(row)+1)

    def on_move_record_up(self):
        rows = self.predictor_tree.selection()
        for row in rows:
            self.predictor_tree.move(row, self.predictor_tree.parent(row), self.predictor_tree.index(row)-1)

    def on_prediction_analysis(self):
        def csv_name_entry_initial_clearer(e, csv_name_entry):
            csv_name_entry.delete(0, END)

        def on_column_clicked_analysis():
            pass

        def on_save_to_csv(prediction_analysis_buttons_frame, tree_all_predicted_values_list, tree_all_actual_values_list):
            def on_save_button():
                prediction_analysis_dataframe.to_csv(csv_name_entry.get() + '.csv', index=False, encoding='utf-8')

            print('tree all predicted values list:', tree_all_predicted_values_list)
            print('tree all actual values list:', tree_all_actual_values_list)
            differences_list = []
            for row in range(len(tree_all_predicted_values_list)):
                print(float(tree_all_predicted_values_list[row]) - float(tree_all_actual_values_list[row]))
                differences_list.append(float(tree_all_predicted_values_list[row]) - float(tree_all_actual_values_list[row]))
            print('differences list:', differences_list)

            dataframe_data = list(zip(tree_all_predicted_values_list, tree_all_actual_values_list, differences_list))

            prediction_analysis_dataframe = pd.DataFrame(dataframe_data, columns=['Predicted Values', 'Actual Values', 'Difference'])
            # Create entry box for csv name
            csv_name_entry = Entry(prediction_analysis_buttons_frame, width=17, font=('Arial', 11, 'italic'))
            csv_name_entry.insert(0, 'Save As')
            save_button = ttk.Button(prediction_analysis_buttons_frame, text='Save', command=on_save_button)

            # Make entry box clear when clicked on
            csv_name_entry.bind('<ButtonRelease-1>', lambda event, csv_name_entry=csv_name_entry: csv_name_entry_initial_clearer(event, csv_name_entry))

            # Locations
            csv_name_entry.grid(row=1, column=0, padx=5)
            save_button.grid(row=2, column=0)

        # Makes sure only one window is possible to open
        global prediction_analysis_window
        try:
            if prediction_analysis_window.state() == 'normal': prediction_analysis_window.focus()
        except:
            # Create window
            prediction_analysis_window = Toplevel()
            prediction_analysis_window.title('Prediction Analysis')
            prediction_analysis_window.geometry('400x350')

            # Create frame for treeview
            prediction_analysis_treeview_frame = Frame(prediction_analysis_window)
            prediction_analysis_treeview_frame.pack(ipadx=200)

            # Create frame for buttons
            prediction_analysis_buttons_frame = Frame(prediction_analysis_window)
            prediction_analysis_buttons_frame.pack(ipadx=200)

            # Create tree
            prediction_analysis_tree = ttk.Treeview(prediction_analysis_treeview_frame)

            # Define columns
            prediction_analysis_tree['columns'] = ('Predicted_Value', 'Actual_Value', 'Difference')

            # Format columns
            prediction_analysis_tree.column('#0', width=0, stretch=NO)
            prediction_analysis_tree.column('Predicted_Value', width=140, stretch=NO)
            prediction_analysis_tree.column('Actual_Value', width=120, stretch=NO)
            prediction_analysis_tree.column('Difference', width=120, stretch=NO)

            # Create headings
            prediction_analysis_tree.heading('Predicted_Value', text='Predicted_Value', anchor=W)
            prediction_analysis_tree.heading('Actual_Value', text='Actual_Value', anchor=W)
            prediction_analysis_tree.heading('Difference', text='Difference', anchor=W)

            # Bind treeview to column click for filter
            prediction_analysis_tree.bind('<Button-1>', on_column_clicked_analysis)

            # Grab values from predictor treeview current selected
            selected = self.predictor_tree.selection()
            tree_all_predicted_values = self.predictor_tree.item(selected, 'values')[10]
            tree_all_actual_values = self.predictor_tree.item(selected, 'values')[11]

            tree_all_predicted_values_list = tree_all_predicted_values.split(', ')
            tree_all_actual_values_list = tree_all_actual_values.split(', ')

            # Create an iteration to grab data and put it into treeview
            index = max_difference = 0
            min_difference = 99999999999999999999999999
            for row in range(len(tree_all_predicted_values_list)):
                prediction_analysis_tree.insert(parent='', index='end', iid=index, text='',
                    values=(str(tree_all_predicted_values_list[row]), str(tree_all_actual_values_list[row]),
                            (float(tree_all_predicted_values_list[row]) - float(tree_all_actual_values_list[row]))))
                individual_difference = abs(float(tree_all_predicted_values_list[row]) - float(tree_all_actual_values_list[row]))
                if individual_difference > max_difference:
                    max_difference = individual_difference
                if individual_difference < min_difference:
                    min_difference = individual_difference
                index += 1


            # Widgets
            save_to_csv_button = ttk.Button(prediction_analysis_buttons_frame, text='Save to CSV',
                    command=lambda: on_save_to_csv(prediction_analysis_buttons_frame, tree_all_predicted_values_list, tree_all_actual_values_list))
            max_difference_label = Label(prediction_analysis_buttons_frame, text='Max Difference: ' + str(max_difference))
            min_difference_label = Label(prediction_analysis_buttons_frame, text='Minimum Difference: ' + str(min_difference))

            # Locations
            prediction_analysis_tree.pack(expand=True, fill=BOTH, padx=5)
            save_to_csv_button.grid(row=0, column=0, padx=5, pady=5)
            max_difference_label.grid(row=0, column=1, padx=5, pady=5)
            min_difference_label.grid(row=1, column=1, padx=5, pady=5)



    def on_save_current_row_order(self):
        # Connect to database
        conn = sqlite3.connect('Predictions_Database')

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
            conn = sqlite3.connect('Predictions_Database')

            # Create cursor
            cursor = conn.cursor()

            # Insert new reordered data into table
            cursor.execute("INSERT INTO predictions_table VALUES (:Date_Predicted, :Target, :Predicted_Value, :Mean_Absolute_Error, :Score, :Cross_Val_Score, :Data_Known, :Model_Used, :Dataframe, :Features_Used, :All_Pickle_Model_Predictions, :Tested_Actual_Values)",
                       {'Date_Predicted': self.predictor_tree.item(record[0], 'values')[0],
                        'Target': self.predictor_tree.item(record[0], 'values')[1],
                        'Predicted_Value': self.predictor_tree.item(record[0], 'values')[2],
                        'Mean_Absolute_Error': self.predictor_tree.item(record[0], 'values')[3],
                        'Score': self.predictor_tree.item(record[0], 'values')[4],
                        'Cross_Val_Score': self.predictor_tree.item(record[0], 'values')[5],
                        'Data_Known': self.predictor_tree.item(record[0], 'values'[6]),
                        'Model_Used': self.predictor_tree.item(record[0], 'values')[7],
                        'Dataframe': self.predictor_tree.item(record[0], 'values')[8],
                        'Features_Used': self.predictor_tree.item(record[0], 'values')[9],
                        'All_Pickle_Model_Predictions': self.predictor_tree.item(record[0], 'values')[10],
                        'Tested_Actual_Values': self.predictor_tree.item(record[0], 'values')[11]})

            # Commit changes
            conn.commit()

            # Close connection
            conn.close()

        # Clear the treeview table
        self.predictor_tree.delete(*self.predictor_tree.get_children())

        # Refresh the database
        self.query_database()

    def on_test_against_file_button(self):
        pass

    def query_database(self):
        if self.filter_current_model_checkbox.instate(['selected']) == True:
            self.filter_model()
            return
        else:
            pass

        # creates a database if one doesn't already exist, otherwise it connects to it
        conn = sqlite3.connect('Predictions_Database')  # creates a database file and puts it in the directory

        # creates a cursor that does all the editing
        cursor = conn.cursor()

        # Create table in database if it doesn't already exist
        cursor.execute("""CREATE TABLE if not exists predictions_table (
                Date_Predicted DATE,
                Target text,
                Predicted_Value real,
                Mean_Absolute_Error real,
                Score real,
                Cross_Val_Score real,
                Data_Known text,
                Model_Used text,
                Dataframe text,
                Features_Used text,
                All_Pickle_Model_Predictions text,
                Tested_Actual_Values text
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
                                                        record[6], record[7], record[8], record[9], record[10], record[11]))
            # Increment counter
            count += 1


class Predictor:
    def predictor_array_cleaner(self, scaled_df, target_variable, scaled_predictor_df):
        df = self.scaled_df[self.selected_features]
        mean_dataframe = pd.DataFrame(df.mean())
        mean_dataframe = mean_dataframe.T
        # plug in our data to the above dataframe
        mean_dataframe.update(scaled_predictor_df)
        mean_dataframe.drop([self.target_variable], axis=1, inplace=True)
        # now turn the above into an array
        finalized_predictor_array = np.array(mean_dataframe)
        return finalized_predictor_array


    def predictor(self):
        # Make selected feature combination a list
        try:
            self.selected_features = self.selected_features.split(', ')
            self.selected_features.append(self.target_variable)
        except:
            pass
        df = self.scaled_df[self.selected_features]

        X = np.array(df.drop([self.target_variable], axis=1), dtype='object')
        y = np.array(df[self.target_variable], dtype='object')

        finalized_predictor_array = Predictor.predictor_array_cleaner(self, self.scaled_df, self.target_variable,
                                                                      self.scaled_data_we_know_df)
        pickle_in = open('saved_training_pickle_models/' + self.selected_training_model + '.pickle', 'rb')
        old_pickled_regression_line = pickle.load(pickle_in)


        runtimes = 10

        current_model_total_score = 0
        pickle_model_total_score = pickle_total_mean_absolute_error = total_pickle_predicted_value = 0
        for i in range(runtimes):
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            MyLinearRegression = linear_model.LinearRegression().fit(X_train, y_train)

            pickle_model_input_prediction = old_pickled_regression_line.predict(finalized_predictor_array)

            pickle_cross_val_score = cross_val_score(old_pickled_regression_line, X, y, cv=10).mean()
            pickle_model_score = old_pickled_regression_line.score(X_test, y_test)
            pickle_model_total_score += pickle_model_score

            self.all_pickle_model_predictions = old_pickled_regression_line.predict(X_test)  # problem line

            pickle_mean_absolute_error = metrics.mean_absolute_error(y_test, self.all_pickle_model_predictions)
            pickle_total_mean_absolute_error += pickle_mean_absolute_error


        self.pickle_predicted_value = pickle_model_input_prediction[0]
        self.pickle_model_average_score = pickle_model_total_score/runtimes
        self.pickle_cross_val_average_score = pickle_cross_val_score/runtimes
        self.pickle_model_average_mae = pickle_total_mean_absolute_error/runtimes

        self.all_actual_values = y_test
        self.all_pickle_model_predictions = self.all_pickle_model_predictions.tolist()

        return self.all_pickle_model_predictions, y_test

    def predictor_plotter(self, all_current_model_predictions, y_test, target_variable):
        sns.set_style('darkgrid')
        plt.figure(figsize=(15, 10))
        plt.scatter(y_test, all_current_model_predictions, c='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3, label='Prediction Line')
        plt.xlabel('Actual Values for ' + self.target_variable)
        plt.ylabel('Predicted_Values for ' + self.target_variable)
        plt.title('Actual Vs Predicted Values')
        plt.legend(loc='upper left', title='X=Actual Values\nY=Predicted Values')
        plt.get_current_fig_manager().window.state('zoomed')


        # Makes the point values shown when hovered over
        mpc.cursor(hover=True)

        plt.show()



if __name__ == '__main__':
    print('You must run this program from the mother predictor app')


