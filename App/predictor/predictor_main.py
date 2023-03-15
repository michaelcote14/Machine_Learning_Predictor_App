import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from PIL import ImageTk, Image
from Step_2_Visualizing.visualization import Grapher

LARGE_FONT = ('Arial', 11, 'bold')
NORMAL_FONT = ('Arial', 9)
SMALL_FONT = ('Arial', 7)


class Machine_Learning_App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)  # sets up Tkinter, doesn't actually call anything in

        tk.Tk.iconbitmap(self,
                         default="")
        tk.Tk.wm_title(self, "Michael's Machine Learning App")

        container = tk.Frame(self)

        # Fill will fill in any space allowed to a pack, expand will go beyond the bounds allowed by a pack
        container.pack(side='top', fill='both', expand=True)
        # Sets the first row/column as 0 and gives a weight to each row/column
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}  # a dictionary of all the frames in your application

        menubar = tk.Menu(container)
        pages_menu = tk.Menu(menubar, tearoff=0)
        pages_menu.add_command(label='Predictor', command=lambda: self.show_frame(PredictorPage))
        pages_menu.add_command(label='Extras', command=lambda: self.show_frame(ExtrasPage))
        menubar.add_cascade(label='Pages', menu=pages_menu)

        tk.Tk.config(self, menu=menubar)  # actually puts the buttons up there

        # List of frames to open
        for F in (PredictorPage, ExtrasPage, PageOne, PageTwo):  # F stands for frame, add in new pages here
            frame = F(container, self)
            frame.config(bg='gray10')  # Makes all the frames gray

            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(PredictorPage)

    def show_frame(self, cont):  # cont stands for container
        frame = self.frames[cont]
        frame.tkraise()  # makes the frame selected come to the front


class PredictorPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Data
        model_options = [
            'Select Model...',
            'Linear Regression',
            'Logistic Regression',
            'KNN',
            'SVD']

        # Labels
        page_title_label = tk.Label(self, text='Predictor', font=LARGE_FONT, bg='#225BBB', anchor=CENTER)

        # Widgets
        self.select_model_combo_box = ttk.Combobox(self, value=model_options, width=21, justify='center',
                                                   font=('Ariel', 11))
        self.select_model_combo_box.current(1)  # Sets the initial box focus
        self.select_train_data_button = ttk.Button(self, text='Select Train Data',
                                                   command=self.on_select_train_data_button, width=30)
        self.select_test_data_button = ttk.Button(self, text='Select Test Data', command=self.on_select_test_data_button, width=30)
        self.target_variable_combo_box = ttk.Combobox(self, width=21, font=('Ariel', 11))
        self.target_variable_combo_box.bind('<<ComboboxSelected>>', self.on_target_variable_combo_box_click)
        clean_button = ttk.Button(self, text='Clean', width=30, command=self.on_clean_button)
        self.graph_button = ttk.Button(self, text='Graph', width=21, command=self.on_graph_button, state=DISABLED)
        self.view_train_dataframe_button = ttk.Button(self, text='View Train Dataframe', width=23, command=self.on_view_train_dataframe_button,
                                           state=DISABLED)
        select_model_button = ttk.Button(self, text='Select Model', width=30, command=self.on_select_model_button)
        predict_button = ttk.Button(self, text='Predict', width=30, command=self.on_predict_button)
        feature_selection_button = ttk.Button(self, text='Select Features', width=30,
                                              command=self.on_select_features_button)
        self.quick_predict_checkbox = ttk.Checkbutton(self, text='      Quick Predict', command=self.on_quick_predict, width=21)
        self.quick_predict_checkbox.state(['disabled'])
        or_label = Label(self, text='OR', width=3, anchor=CENTER, bg='gray10', fg='white')

        # Locations
        page_title_label.grid(sticky='EW', row=0, column=0, ipadx=40, columnspan=2)
        self.select_model_combo_box.grid(sticky=W, row=1, column=0)
        self.select_train_data_button.grid(sticky=W, row=2, column=0, pady=(5))
        self.select_test_data_button.grid(sticky=W, row=3, column=0, pady=(5))
        or_label.grid(sticky=W, row=3, column=0, pady=5, padx=(195, 0))
        self.quick_predict_checkbox.grid(sticky=W, row=3, column=0, pady=5, padx=(230, 0))
        self.target_variable_combo_box.grid(sticky=W, row=4, column=0, pady=5)
        clean_button.grid(sticky=W, row=5, column=0, pady=5)
        feature_selection_button.grid(sticky=W, row=6, column=0, pady=5)
        select_model_button.grid(sticky=W, row=7, column=0, pady=5)
        self.view_train_dataframe_button.grid(sticky=W, row=1, column=0, pady=5, padx=(230, 0))
        self.graph_button.grid(sticky=W, row=8, column=0, padx=(230, 0))
        predict_button.grid(sticky=W, row=8, column=0, pady=5)


        # Images
        self.green_check_label_model = self.image_creator('../../Images/green_checkmark.png', 24, 24)
        self.green_check_label_train_data = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_test_data = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_target = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_clean = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_features = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_trainer = self.image_creator('../../Images/red_x.png', 24, 24)

        # Image Locations
        self.green_check_label_model.grid(row=1, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_train_data.grid(row=2, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_test_data.grid(row=3, column=0, sticky=W, padx=(383, 10))
        self.green_check_label_target.grid(row=4, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_clean.grid(row=5, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_features.grid(row=6, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_trainer.grid(row=7, column=0, sticky=W, padx=(195, 10))

    def feature_combination_chooser(self):
        FeatureCombinationPage(self.target_variable, self.selected_important_features, self.scaled_train_df, self.data_name,
                               self.update_feature_combination)

    def image_changer(self, img_location, label_to_configure, width, height):
        red_x_image = Image.open(img_location)
        new_size = red_x_image.resize((width, height), Image.Resampling.LANCZOS)
        my_img2 = ImageTk.PhotoImage(new_size)
        label_to_configure.configure(image=my_img2, text='green')
        label_to_configure.ImageTk = my_img2

    def image_creator(self, img_location, width, height, bg='gray10'):
        green_check_image = Image.open(img_location)
        new_size = green_check_image.resize((width, height), Image.Resampling.LANCZOS)
        new_img = ImageTk.PhotoImage(new_size)
        label_image = Label(self, image=new_img, bg=bg, text='red')
        label_image.ImageTk = new_img
        return label_image

    def important_features_selector(self):
        important_features_object = ImportantFeaturesPage(self.scaled_train_df, self.target_variable,
                                                          self.update_important_features, self.data_name)

    def on_clean_button(self):
        from Step_1_Data_Cleaning.data_cleaner import full_cleaner, test_df_empty_row_averager
        # Get the average train_df for any unknown values in test_df
        self.filled_test_df = test_df_empty_row_averager(self.train_df, self.test_df)


        print('Train_df skew scores:')
        self.fully_cleaned_train_df, columns_removed, rows_removed = full_cleaner(self.train_df, self.target_variable)
        print('\nTest_df skew scores:')
        self.fully_cleaned_test_df, columns_removed, rows_removed = full_cleaner(self.filled_test_df)

        # A popup info box that shows how many columns and rows were removed from cleaning
        messagebox.showinfo('Cleaning Results', 'CLEANING RESULTS:' +
                            '\n\nCOLUMNS REMOVED: ' + str(columns_removed) + '\n\nTOTAL ROWS REMOVED: ' + str(
            rows_removed))

        from Step_3_Single_Encoding.single_hot_encoder import single_encoder
        single_encoded_train_df, single_encoded_test_df = single_encoder(self.fully_cleaned_train_df, self.fully_cleaned_test_df)

        from Step_4_Multiple_Encoding.multiple_hot_encoder import multiple_encoder
        self.multiple_encoded_train_df, self.multiple_encoded_test_df = multiple_encoder(self.fully_cleaned_train_df, single_encoded_train_df,
                                                                               self.fully_cleaned_test_df,
                                                                               single_encoded_test_df)

        # Drop the target variable from multiple encoded df
        target_column = self.multiple_encoded_train_df[self.target_variable]
        self.multiple_encoded_train_df.drop(self.target_variable, axis=1, inplace=True)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.multiple_encoded_train_df)
        self.scaled_train_df = pd.DataFrame(scaler.transform(self.multiple_encoded_train_df), index=self.multiple_encoded_train_df.index, columns=self.multiple_encoded_train_df.columns)
        scaler.fit(self.multiple_encoded_test_df)
        self.scaled_test_df = pd.DataFrame(scaler.transform(self.multiple_encoded_test_df), index=self.multiple_encoded_test_df.index, columns=self.multiple_encoded_test_df.columns)

        # Add the target variable back to the dataframe
        self.scaled_train_df[self.target_variable] = target_column

        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_clean['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)

        # Changes the red x to a green checkmark
        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_clean, 24, 24)

    def on_graph_button(self):
        Grapher.main_visualizer(self, self.train_df, self.target_variable)

    def on_predict_button(self):
        # Add the target variable column back to the multiple encoded df
        self.multiple_encoded_train_df = pd.concat([self.multiple_encoded_train_df, self.train_df[self.target_variable]], axis=1)

        from Step_10_Predicting.predictor import PredictorTreeviewPage
        PredictorTreeviewPage(self.scaled_train_df, self.target_variable, self.data_name, self.test_df, self.scaled_test_df)

    def on_quick_predict(self):
        def clear_quick_predict_entry_box(event):
            quick_predict_entry.delete(0, END)
        def on_predict():
            quick_predict_selected_features = []
            for combo_box in self.quick_predict_combo_box_list:
                if combo_box.get() != 'Select Feature':
                    quick_predict_selected_features.append(combo_box.get())

            quick_predict_selected_values = []
            for entry_box in self.quick_predict_entry_boxes_list:
                if entry_box.get() != '':
                    quick_predict_selected_values.append(entry_box.get())

            # Create a dictionary of the selected features and values
            quick_predict_selected_features_and_values = dict(zip(quick_predict_selected_features, quick_predict_selected_values))

            # Create a dataframe of the selected features and values
            self.test_df = pd.DataFrame(quick_predict_selected_features_and_values, index=[0])

            # Make sure all the numeric columns are actually considered numeric
            for column in self.test_df.columns:
                if self.test_df[column].dtype == 'object':
                    try:
                        self.test_df[column] = pd.to_numeric(self.test_df[column])
                    except:
                        pass

            # Close the quick predict window
            quick_predict_window.destroy()

            self.image_changer('../../Images/green_checkmark.png', self.green_check_label_test_data, 24, 24)

            # Checks the status of the green check image and changes images after button click
            if self.green_check_label_test_data['text'] == 'green':
                self.image_changer('../../Images/red_x.png', self.green_check_label_target, 24, 24)
                self.image_changer('../../Images/red_x.png', self.green_check_label_clean, 24, 24)
                self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
                self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)


        # Check the state of the quick predict checkbutton
        if self.quick_predict_checkbox.instate(['selected']) == True:
            pass
        else:
            # Change the state of the quick predict checkbutton
            self.quick_predict_checkbox.state(['!selected'])

            # Enable the select test data button
            self.select_test_data_button.config(state='normal')

            self.image_changer('../../Images/red_x.png', self.green_check_label_test_data, 24, 24)

            return

        # Disable the select test data button
        self.select_test_data_button.config(state='disabled')

        # Makes sure only one window opens
        global quick_predict_window
        try:
            if quick_predict_window.state() == 'normal': quick_predict_window.focus()
        except:
            # Create window
            quick_predict_window = Toplevel()
            quick_predict_window.title('Quick Predict')
            quick_predict_window.geometry('800x120')

            # Create a main frame
            quick_predict_frame = Frame(quick_predict_window)
            quick_predict_frame.pack(fill=BOTH, expand=True)

            # Create a canvas for the frame
            quick_predict_canvas = Canvas(quick_predict_frame, width=800, height=120)
            quick_predict_canvas.pack(side='top', fill='both', expand=True)

            # Add a scrollbar to the canvas
            quick_predict_window_scrollbar = ttk.Scrollbar(quick_predict_canvas, orient=HORIZONTAL,
                                            command=quick_predict_canvas.xview)
            quick_predict_window_scrollbar.pack(side=BOTTOM, fill=X)

            # Configure the canvas
            quick_predict_canvas.configure(xscrollcommand=quick_predict_window_scrollbar.set)
            quick_predict_canvas.bind('<Configure>', lambda e: quick_predict_canvas.configure(scrollregion=quick_predict_canvas.bbox('all')))

            # Create a second frame inside the canvas where all the widgets go
            second_quick_predict_frame = Frame(quick_predict_canvas)

            # Add that new frame to a window in the canvas
            quick_predict_canvas.create_window((0, 0), window=second_quick_predict_frame, anchor='nw')

            # Widgets
            predict_button = ttk.Button(second_quick_predict_frame, text='Finished', width = 30, command=on_predict)

            self.quick_predict_combo_box_list = []
            self.quick_predict_entry_boxes_list = []
            for index in range(0, len(self.train_df.columns.tolist())):
                quick_predict_combo_box = ttk.Combobox(second_quick_predict_frame, values=self.train_df.columns.tolist(), width=17)
                quick_predict_combo_box.set('Select Feature')
                quick_predict_entry = Entry(second_quick_predict_frame, width=20)

                self.quick_predict_combo_box_list.append(quick_predict_combo_box)
                self.quick_predict_entry_boxes_list.append(quick_predict_entry)

                quick_predict_entry.bind('<Button-1>', lambda event: clear_quick_predict_entry_box(event))

                # Locations
                quick_predict_combo_box.grid(row=0, column=0 + index, padx=(0, 10))
                quick_predict_entry.grid(row=1, column=0 + index, padx=(0, 10))
                predict_button.grid(row=2, column=2, pady=(10, 0), columnspan=2)

    def on_select_features_button(self):
        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_features['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)

        from Step_7_Feature_Importance_Finding.feature_importance_finder import MostImportantFeaturesPage
        MostImportantFeaturesPage(self.target_variable, self.scaled_train_df, self.data_name, self.image_changer,
                                  self.green_check_label_features)

    def on_select_train_data_button(self):
        self.train_df_location = filedialog.askopenfilename(initialdir='/', title='Select A CSV or Pickle File',
                                                            filetypes=(('pickle files', '*.pickle'), ('csv files', '*.csv')))
        self.data_name = self.train_df_location[self.train_df_location.rfind('/', 0) + 1:]

        # Grabs the correct format of the data
        if self.train_df_location.endswith('.csv'):
            self.train_df = pd.read_csv(self.train_df_location)
        else:
            self.train_df = pd.read_pickle(self.train_df_location)

        self.target_variable_options = self.train_df.columns.tolist()

        filtered_target_variable_list = []
        if self.select_model_combo_box.get() == 'Linear Regression' or self.select_model_combo_box.get() == 'Logistic Regression':
            [filtered_target_variable_list.append(column) for column in self.train_df.columns if
             self.train_df[column].dtype == 'float64' or self.train_df[column].dtype == 'int64']
        elif self.select_model_combo_box.get() == 'KNN' or self.select_model_combo_box.get() == 'SVD':
            [filtered_target_variable_list.append(column) for column in self.train_df.columns if
             self.train_df[column].dtype != 'float64' or self.train_df[column].dtype == 'int64']

        # Sort the target variable list
        filtered_target_variable_list = sorted(filtered_target_variable_list, key=str.casefold)
        filtered_target_variable_list.insert(0, 'Select Target Variable...')

        self.target_variable_combo_box.config(value=filtered_target_variable_list, width=22, justify='center',
                                              state=NORMAL)

        self.target_variable_combo_box.set(filtered_target_variable_list[0])
        self.select_model_combo_box.bind('<<ComboboxSelected>>', self.select_model_combo_click)

        # Changes the button text
        self.select_train_data_button.config(text=self.data_name)

        # Enables the quick predict check button
        self.quick_predict_checkbox.config(state='normal')
        # Make the quick predict checkbox unchecked by default
        self.quick_predict_checkbox.state(['!alternate'])

        # Enables the view data button
        self.view_train_dataframe_button.config(state=NORMAL)

        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_train_data, 24, 24)

        # Checks the status of the green check image and changes images after button click        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_train_data['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_test_data, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_target, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_clean, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)

    def on_select_test_data_button(self):
        self.test_df_location = filedialog.askopenfilename(initialdir='/', title='Select A CSV File',
                                                           filetypes=(('pickle files', '*.pickle'), ('csv files', '*.csv')))
        self.test_df_name = self.test_df_location[
                                   self.test_df_location.rfind('/', 0) + 1:]

        # Changes the button text
        self.select_test_data_button.config(text=self.test_df_name)

        # Grabs the correct format of the data
        if self.test_df_location.endswith('.csv'):
            self.test_df = pd.read_csv(self.test_df_location, encoding='utf-8')
        else:
            self.test_df = pd.read_pickle(self.test_df_location)


        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_test_data, 24, 24)

        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_test_data['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_target, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_clean, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)

    def on_target_variable_combo_box_click(self, event):
        self.target_variable = self.target_variable_combo_box.get()

        # Makes the graph button clickable
        self.graph_button.config(state=NORMAL)

        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_target['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_clean, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)

        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_target, 24, 24)

    def on_select_model_button(self):
        from Step_8_Training.trainer import TrainingModelPage
        TrainingModelPage(self.scaled_train_df, self.target_variable, self.data_name)

        # Changes the red x to a green checkmark
        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_trainer, 24, 24)

    def on_view_train_dataframe_button(self):
        ViewDataPage(self.train_df_location)

    def select_model_combo_click(self, event):
        filtered_target_variable_list = []
        if select_model_combo_box.get() == 'Linear Regression' or select_model_combo_box.get() == 'Logistic Regression':
            [filtered_target_variable_list.append(column) for column in self.train_df.columns if
             self.train_df[column].dtype == 'int64']
        elif select_model_combo_box.get() == 'KNN' or select_model_combo_box.get() == 'SVD':
            [filtered_target_variable_list.append(column) for column in self.train_df.columns if
             self.train_df[column].dtype != 'int64']

        self.target_variable_combo_box.configure(values=filtered_target_variable_list)

    def trainer_runtime_entry_clearer(self, e):
        self.trainer_runtimes_entry.delete(0, END)


class ViewDataPage:
    def __init__(self, data_location):
        # Makes sure that the window will only be opened once
        global data_viewer_window
        try:
            if data_viewer_window.state() == 'normal': data_viewer_window.focus()
        except:
            data_viewer_window = Toplevel()
            # ToDo put in view test dataframe button

            if str(data_location).endswith('.csv'):
                self.dataframe_to_view = pd.read_csv(data_location)
            else:
                self.dataframe_to_view = pd.read_pickle(data_location)

            data_viewer_window.title('Data Viewer')
            data_viewer_window.geometry('1200x300')

            # Create frame
            data_viewer_frame = Frame(data_viewer_window, bg='gray10')
            data_viewer_frame.pack()

            # Create treeview
            self.data_viewer_tree = ttk.Treeview(data_viewer_frame)

            # Configure scrollbar
            view_train_dataframe_horizontal_scrollbar = ttk.Scrollbar(data_viewer_frame, orient=HORIZONTAL,
                                                           command=self.data_viewer_tree.xview)
            view_train_dataframe_horizontal_scrollbar.pack(side=BOTTOM, fill=X)
            self.data_viewer_tree.configure(xscrollcommand=view_train_dataframe_horizontal_scrollbar.set)

            # Set up new tree view
            self.data_viewer_tree['columns'] = list(self.dataframe_to_view.columns)
            self.data_viewer_tree['show'] = 'headings'

            # Loop through column list to create the tree headers
            for column in self.data_viewer_tree['columns']:
                self.data_viewer_tree.heading(column, text=column, anchor=W)

            # Put data in treeview
            self.df_rows = self.dataframe_to_view.to_numpy().tolist()
            for row in self.df_rows:
                self.data_viewer_tree.insert('', 'end', values=row)

            # Bind treeview with left mouse-click
            self.data_viewer_tree.bind('<Button-1>', self.on_column_clicked)
            self.sorted_state = 'off'

            self.data_viewer_tree.pack(padx=5)

    def on_column_clicked(self, event):
        region_clicked = self.data_viewer_tree.identify_region(event.x, event.y)

        if region_clicked not in 'heading':
            return
        if self.sorted_state == 'off':
            # How to identify which column was clicked
            column_clicked = self.data_viewer_tree.identify_column(event.x)
            column_clicked_index = int(column_clicked[1:]) - 1

            # Puts a down arrow in the column name
            self.data_viewer_tree.heading(list(self.dataframe_to_view)[column_clicked_index],
                                          text=list(self.dataframe_to_view)[column_clicked_index] + ' ' * 8 + 'V')

            self.sorted_state = 'on'

            column_clicked_name = self.data_viewer_tree['columns'][column_clicked_index]
            # sorts the original pandas dataframe
            sorted_dataframe = self.dataframe_to_view.sort_values(by=column_clicked_name, ascending=False)

            # Clear old treeview
            self.data_viewer_tree.delete(*self.data_viewer_tree.get_children())

            # Put sorted dataframe in treeview
            sorted_array = sorted_dataframe.to_numpy().tolist()
            for row in sorted_array:
                self.data_viewer_tree.insert('', 'end', values=row)

        else:
            # Reload the original treeview data
            for column in self.data_viewer_tree['columns']:
                self.data_viewer_tree.heading(column, text=column)

            self.sorted_state = 'off'


class ExtrasPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = Label(self, text='Extras Page', font=LARGE_FONT)
        label.pack(padx=10, pady=10)


class PageOne(tk.Frame):
    # Pretty much entire page goes under init method
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Page 1', font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        # How to pass variables to a command
        button1 = tk.Button(self, text='Back to Home', command=lambda: controller.show_frame(Predictor))
        button1.pack()

        button2 = tk.Button(self, text='Page 2', command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageTwo(tk.Frame):
    # Pretty much entire page goes under init method
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text='Page 2', font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        # How to pass variables to a command
        button1 = tk.Button(self, text='Back to Home', command=lambda: controller.show_frame(Predictor))
        button1.pack()

        button2 = tk.Button(self, text='Page 1', command=lambda: controller.show_frame(PageOne))
        button2.pack()


app = Machine_Learning_App()
app.geometry('415x315')
app.resizable(False, False)

app.mainloop()
