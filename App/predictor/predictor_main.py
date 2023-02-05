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
        spinbox_variable = StringVar(self)
        self.scaler_runtimes_spinbox = Spinbox(self, from_=500, to=100000, increment=500, textvariable=spinbox_variable,
                                               font=('Ariel', 13, 'bold'), width=13)
        spinbox_variable.set('Runtimes')
        self.graph_button = ttk.Button(self, text='Graph', width=21, command=self.on_graph_button, state=DISABLED)
        self.view_data_button = ttk.Button(self, text='View Data', width=21, command=self.on_view_data_button,
                                           state=DISABLED)
        select_model_button = ttk.Button(self, text='Select Model', width=30, command=self.on_train_model_button)
        predict_button = ttk.Button(self, text='Predict', width=30, command=self.on_predict_button)
        feature_selection_button = ttk.Button(self, text='Select Features', width=30,
                                              command=self.on_select_features_button)
        scale_button = ttk.Button(self, text='Scale', command=self.on_scale_button, width=7)

        # Images
        self.green_check_label_model = self.image_creator('../../Images/green_checkmark.png', 24, 24)
        self.green_check_label_train_data = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_test_data = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_target = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_clean = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_scale = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_features = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_trainer = self.image_creator('../../Images/red_x.png', 24, 24)
        self.green_check_label_predict = self.image_creator('../../Images/red_x.png', 24, 24)

        # Image Locations
        self.green_check_label_model.grid(row=1, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_train_data.grid(row=2, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_test_data.grid(row=3, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_target.grid(row=4, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_clean.grid(row=5, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_scale.grid(row=6, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_features.grid(row=7, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_trainer.grid(row=8, column=0, sticky=W, padx=(195, 10))
        self.green_check_label_predict.grid(row=9, column=0, sticky=W, padx=(195, 10))

        # Locations
        page_title_label.grid(sticky='EW', row=0, column=0, ipadx=40, columnspan=2)
        self.select_model_combo_box.grid(sticky=W, row=1, column=0)
        self.select_train_data_button.grid(sticky=W, row=2, column=0, pady=(5))
        self.select_test_data_button.grid(sticky=W, row=3, column=0, pady=(5))
        self.target_variable_combo_box.grid(sticky=W, row=4, column=0, pady=5)
        clean_button.grid(sticky=W, row=5, column=0, pady=5)
        scale_button.grid(sticky=W, row=6, column=0, pady=5, padx=(139, 0))
        self.scaler_runtimes_spinbox.grid(sticky=W, row=6, column=0, pady=5)
        feature_selection_button.grid(sticky=W, row=7, column=0, pady=5)
        select_model_button.grid(sticky=W, row=8, column=0, pady=5)
        self.graph_button.grid(sticky=W, row=9, column=0, padx=(295, 0))
        self.view_data_button.grid(sticky=W, row=1, column=0, pady=5, padx=(295, 0))
        predict_button.grid(sticky=W, row=9, column=0, pady=5)

    def feature_combination_chooser(self):
        FeatureCombinationPage(self.target_variable, self.selected_important_features, self.scaled_df, self.csv_name,
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
        important_features_object = ImportantFeaturesPage(self.scaled_df, self.target_variable,
                                                          self.update_important_features, self.csv_name)

    def on_clean_button(self):
        # Drop target variable from original df
        self.non_target_original_df = self.original_df.drop([self.target_variable], axis=1)

        from Step_1_Data_Cleaning.data_cleaner import full_cleaner
        self.fully_cleaned_df, columns_removed, rows_removed = full_cleaner(self.original_df, self.target_variable)
        self.fully_cleaned_df2, columns_removed, rows_removed = full_cleaner(self.original_df2)

        # A popup info box that shows how many columns and rows were removed from cleaning
        messagebox.showinfo('Cleaning Results', 'CLEANING RESULTS:' +
                            '\n\nCOLUMNS REMOVED: ' + str(columns_removed) + '\n\nTOTAL ROWS REMOVED: ' + str(
            rows_removed))

        from Step_3_Single_Encoding.single_hot_encoder import single_encoder
        single_encoded_df, single_encoded_df2 = single_encoder(self.fully_cleaned_df, self.fully_cleaned_df2)

        from Step_4_Multiple_Encoding.multiple_hot_encoder import multiple_encoder
        self.multiple_encoded_df, self.multiple_encoded_df2 = multiple_encoder(self.fully_cleaned_df, single_encoded_df,
                                                                               self.fully_cleaned_df2,
                                                                               single_encoded_df2)

        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_clean['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_scale, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_predict, 24, 24)

        # Changes the red x to a green checkmark
        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_clean, 24, 24)

    def on_graph_button(self):
        Grapher.main_visualizer(self, self.original_df, self.target_variable)

    def on_predict_button(self):
        # Add the target variable column back to the multiple encoded df
        self.multiple_encoded_df = pd.concat([self.multiple_encoded_df, self.original_df[self.target_variable]], axis=1)

        from Step_10_Predicting.predictor import PredictorTreeviewPage
        PredictorTreeviewPage(self.scaled_df, self.target_variable, self.csv_name, self.original_df2, self.scaled_df2)


        # Changes the red x to a green checkmark
        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_predict, 24, 24)

    def on_scale_button(self):
        scaler_runtimes = int(self.scaler_runtimes_spinbox.get())

        from Step_5_Scaling.scaler import Scaler
        scaler_predicted_time = Scaler.scaler_time_predictor(self, scaler_runtimes, self.multiple_encoded_df)
        scaler_response = messagebox.askyesno('Caution', 'Scaler will take about\n' + str(scaler_predicted_time) +
                                              '\nto run, are you sure you want to continue?')
        if scaler_response == True:
            pass
        else:
            return

        scaler_progressbar = ttk.Progressbar(self, orient=HORIZONTAL, length=100, mode='determinate')
        scaler_progressbar.grid(row=6, column=0, padx=(120, 0))

        # ToDo fix the scaling situation
        # Drop the target variable from multiple encoded df
        target_column = self.multiple_encoded_df[self.target_variable]
        self.multiple_encoded_df.drop(self.target_variable, axis=1, inplace=True)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.multiple_encoded_df)
        self.scaled_df = pd.DataFrame(scaler.transform(self.multiple_encoded_df), index=self.multiple_encoded_df.index, columns=self.multiple_encoded_df.columns)
        scaler.fit(self.multiple_encoded_df2)
        self.scaled_df2 = pd.DataFrame(scaler.transform(self.multiple_encoded_df2), index=self.multiple_encoded_df2.index, columns=self.multiple_encoded_df2.columns)

        # Add the target variable back to the dataframe
        self.scaled_df[self.target_variable] = target_column

        # self.scaled_df, self.scaled_df2 = Scaler.main_scaler(self, scaler_runtimes,
        #                                                      self.target_variable, scaler_progressbar, self.master,
        #                                                      self.multiple_encoded_df, self.multiple_encoded_df2)
        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_scale['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_predict, 24, 24)

        # Changes the red x to a green checkmark
        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_scale, 24, 24)

    def on_select_features_button(self):
        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_features['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_predict, 24, 24)

        from Step_7_Feature_Combination_Testing.feature_selection import FeatureSelectionPage
        FeatureSelectionPage(self.target_variable, self.scaled_df, self.csv_name, self.image_changer,
                             self.green_check_label_features)

    def on_select_train_data_button(self):
        self.csv_train_location = filedialog.askopenfilename(initialdir='/', title='Select A CSV File',
                                                             filetypes=(('csv files', '*.csv'),))
        self.csv_name = self.csv_train_location[self.csv_train_location.rfind('/', 0) + 1:]

        # Changes the button text
        self.select_train_data_button.config(text=self.csv_name)

        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_train_data, 24, 24)

        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_train_data['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_test_data, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_target, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_clean, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_scale, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_predict, 24, 24)

    def on_select_test_data_button(self):
        self.csv_data_we_know_location = filedialog.askopenfilename(initialdir='/', title='Select A CSV File',
                                                                    filetypes=(('csv files', '*.csv'),))
        self.csv_data_we_know_name = self.csv_data_we_know_location[
                                     self.csv_data_we_know_location.rfind('/', 0) + 1:]

        # Changes the button text
        self.select_test_data_button.config(text=self.csv_data_we_know_name)

        self.original_df = pd.read_csv(self.csv_train_location)
        self.original_df2 = pd.read_csv(self.csv_data_we_know_location, encoding='utf-8')
        self.csv_location = self.csv_train_location

        self.target_variable_options = self.original_df.columns.tolist()

        filtered_target_variable_list = []
        if self.select_model_combo_box.get() == 'Linear Regression' or self.select_model_combo_box.get() == 'Logistic Regression':
            [filtered_target_variable_list.append(column) for column in self.original_df.columns if
             self.original_df[column].dtype == 'float64' or self.original_df[column].dtype == 'int64']
        elif self.select_model_combo_box.get() == 'KNN' or self.select_model_combo_box.get() == 'SVD':
            [filtered_target_variable_list.append(column) for column in self.original_df.columns if
             self.original_df[column].dtype != 'float64' or self.original_df[column].dtype == 'int64']

        # Sort the target variable list
        filtered_target_variable_list = sorted(filtered_target_variable_list, key=str.casefold)
        filtered_target_variable_list.insert(0, 'Select Target Variable...')

        self.target_variable_combo_box.config(value=filtered_target_variable_list, width=22, justify='center',
                                              state=NORMAL)
        self.target_variable_combo_box.set(filtered_target_variable_list[0])
        self.select_model_combo_box.bind('<<ComboboxSelected>>', self.select_model_combo_click)
        self.view_data_button.config(state=NORMAL)
        try:
            self.select_data_button.config(text=self.csv_name)
        except:
            pass

        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_test_data, 24, 24)

        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_test_data['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_target, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_clean, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_scale, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_predict, 24, 24)

    def on_target_variable_combo_box_click(self, event):
        self.target_variable = self.target_variable_combo_box.get()

        # Makes the graph button clickable
        self.graph_button.config(state=NORMAL)

        # Checks the status of the green check image and changes images after button click
        if self.green_check_label_target['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_clean, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_scale, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_features, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_trainer, 24, 24)
            self.image_changer('../../Images/red_x.png', self.green_check_label_predict, 24, 24)

        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_target, 24, 24)

    def on_train_model_button(self):
        # Checks the status of the green check image and changes images after button click

        from Step_8_Training.trainer import TrainingModelPage
        TrainingModelPage(self.scaled_df, self.target_variable, self.csv_name)

        if self.green_check_label_trainer['text'] == 'green':
            self.image_changer('../../Images/red_x.png', self.green_check_label_predict, 24, 24)

        # Changes the red x to a green checkmark
        self.image_changer('../../Images/green_checkmark.png', self.green_check_label_trainer, 24, 24)

    def on_view_data_button(self):
        ViewDataPage(self.csv_location, 'viewing')

    def scalar_runtime_entry_clearer(self, e):
        self.scaler_runtimes_entry.delete(0, END)

    def select_model_combo_click(self, event):
        filtered_target_variable_list = []
        if select_model_combo_box.get() == 'Linear Regression' or select_model_combo_box.get() == 'Logistic Regression':
            [filtered_target_variable_list.append(column) for column in self.original_df.columns if
             self.original_df[column].dtype == 'int64']
        elif select_model_combo_box.get() == 'KNN' or select_model_combo_box.get() == 'SVD':
            [filtered_target_variable_list.append(column) for column in self.original_df.columns if
             self.original_df[column].dtype != 'int64']

        self.target_variable_combo_box.configure(values=filtered_target_variable_list)

    def trainer_runtime_entry_clearer(self, e):
        self.trainer_runtimes_entry.delete(0, END)


class ViewDataPage:
    def __init__(self, csv_location):
        # Makes sure that the window will only be opened once
        global data_viewer_window
        try:
            if data_viewer_window.state() == 'normal': data_viewer_window.focus()
        except:
            data_viewer_window = Toplevel()
            self.dataframe_to_view = pd.read_csv(csv_location)
            data_viewer_window.title('Data Viewer')
            data_viewer_window.geometry('1200x300')

            # Create frame
            data_viewer_frame = Frame(data_viewer_window)
            data_viewer_frame.pack()

            # Create treeview
            self.data_viewer_tree = ttk.Treeview(data_viewer_frame)

            # Configure scrollbar
            view_data_horizontal_scrollbar = ttk.Scrollbar(data_viewer_frame, orient=HORIZONTAL,
                                                           command=self.data_viewer_tree.xview)
            view_data_horizontal_scrollbar.pack(side=BOTTOM, fill=X)
            self.data_viewer_tree.configure(xscrollcommand=view_data_horizontal_scrollbar.set)

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
app.geometry('516x430')
app.resizable(False, False)

app.mainloop()
