from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
from Step_1_Visualizing import visualization

# ToDo put in graphs
# ToDo add in focus for the scalar runtimes box

def view_csv():
    from machine_learning_app.predictor import treeview_csv_viewer
    treeview_csv_viewer.file_open()
    print('View CSV Button Clicked')
    # ToDo figure out why the view csv treeviewer isn't working
    csv_viewer_window = Tk()
    csv_viewer_window.title('CSV Tree View')
    csv_viewer_window.geometry('1200x300')

    # Create frame
    csv_viewer_frame = Frame(csv_viewer_window)
    csv_viewer_frame.pack()

    # Create treeview
    csv_viewer_tree = ttk.Treeview(csv_viewer_frame)

    # Create Open File Button
    open_file_button = Button(csv_viewer_frame, text='Open File', command=file_open)
    open_file_button.pack()

root = Tk()
root.title("Michael's Machine Learning App")
root.geometry('1500x900')
root.config(bg='gray17')
colors = {'blue': '#1E90FF', 'gray': 'gray17'}

my_notebook = ttk.Notebook(root)
my_notebook.pack()

# puts the first frame in the notebook
models_tester_frame = Frame(my_notebook, width=1500, height=900, bg='gray17')
model_runner_frame = Frame(my_notebook, width=1500, height=900, bg='gray17')
log_history_frame = Frame(my_notebook, width=1500, height=900, bg='gray17')
saved_data_frame = Frame(my_notebook, width=1500, height=900, bg='gray17')
cleaning_tester_frame = Frame(my_notebook, width=1500, height=900, bg='gray17')

# fill makes the frame expand to the full size of the window, expand will make it the size of the initial window
models_tester_frame.pack(fill='both', expand=1)
model_runner_frame.pack(fill='both', expand=1)
log_history_frame.pack(fill='both', expand=1)
saved_data_frame.pack(fill='both', expand=1)
cleaning_tester_frame.pack(fill='both', expand=1)


# creates the tabs and names them
my_notebook.add(models_tester_frame, text='Models Tester')
my_notebook.add(model_runner_frame, text='Model Runner')
my_notebook.add(log_history_frame, text='Log History')
my_notebook.add(saved_data_frame, text='Saved Data')
my_notebook.add(cleaning_tester_frame, text='Cleaning Tester')

################## Models Tester Frame ####################################
# functions
def checkbox_grabber():
    for i in range(len(models)):
        selected = ''
        if models[i].get()>=1:
            selected += str(i)
            messagebox.showinfo(message="You selected Checkbox " + selected)
        chosen_models.append(selected)
    print(chosen_models)


chosen_model = IntVar()
chosen_model.set('')

models = []
chosen_models = []

# labels
model_chooser_label = Label(models_tester_frame, text='Select Which Models to Test', font=('Helvetica', 12), padx=6)

for i in range(5): # choose how many checkboxes you will have plus 1
    chosen_model = IntVar()
    chosen_model.set(0)
    models.append(chosen_model)

# buttons
# the onvalue changes what the selected box will return. Usually it will return a 1 if selected and a 0 if not selected
linear_checkbox_button = Checkbutton(models_tester_frame, text='Linear Regression', variable=models[1], padx=48)
logistic_checkbox_button = Checkbutton(models_tester_frame, text='Logistic Regression', variable=models[2], padx=43)
KNN_checkbox_button = Checkbutton(models_tester_frame, text='KNN', variable=models[3], padx=81)
SVD_checkbox_button = Checkbutton(models_tester_frame, text='SVD', variable=models[4], padx=83)
checkbox_submit_button = Button(models_tester_frame, text='Run Tester', command=checkbox_grabber, padx=48)

# locations
model_chooser_label.grid(row=0, column=3)
linear_checkbox_button.grid(row=1, column=3)
logistic_checkbox_button.grid(row=2, column=3)
KNN_checkbox_button.grid(row=3, column=3)
SVD_checkbox_button.grid(row=4, column=3)
checkbox_submit_button.grid(row=5, column=3)


######################### Model Runner Frame #######################################
data_known_frame = Frame(model_runner_frame)
# Functions
def app_csv_opener():
    # Initialization
    global csv_location
    csv_location = filedialog.askopenfilename(initialdir='/', title='Select A CSV File',
    filetypes=(('csv files', '*.csv'), ('all files', '*.*')))
    global csv_name
    csv_name = csv_location[csv_location.rfind('/',0) + 1:]

    # Data
    global original_df
    original_df = pd.read_csv(csv_location)
    target_variable_options = original_df.columns.tolist()

    # Deletes the data known frame when data is selected more than once
    global data_known_frame
    data_known_frame.destroy()

    # Widgets
    global target_variable_combo_box
    target_variable_combo_box = ttk.Combobox(model_runner_frame, value=target_variable_options, width=20)
    target_variable_combo_box.set(target_variable_options[0])
    target_variable_combo_box.bind('<<ComboboxSelected>>', target_variable_combo_click)
    data_known_frame = Frame(model_runner_frame, highlightbackground='red', highlightthickness=2, bg='gray17')
    csv_viewer_button = Button(model_runner_frame, text='View CSV', bg='green', command=view_csv)

    # Grabs all of the data from the data we know section
    global values_we_know_list
    global features_we_know_list
    values_we_know_list = []
    features_we_know_list = []
    combo_box_list = original_df.columns.tolist()
    combo_box_list.insert(0, "")
    for i in range(len(original_df.columns)):
        features_we_know_combo_box = ttk.Combobox(data_known_frame, values=combo_box_list, width=17)
        features_we_know_combo_box.grid_forget()
        features_we_know_combo_box.grid(row=1, column=i+2)
        global entry_boxes
        data_we_know_entry_box = Entry(data_known_frame)
        data_we_know_entry_box.grid_forget()
        data_we_know_entry_box.grid(row=2, column=i+2, padx=5, sticky=N)
        values_we_know_list.append(data_we_know_entry_box)
        features_we_know_list.append(features_we_know_combo_box)



    # labels
    selected_data_label = Label(model_runner_frame, text=csv_name,
                bg='gray17', fg='white')
    data_known_label = Label(data_known_frame, text='Data Known', width=160, bg='white')

    # locations
    data_known_frame.grid_forget()
    data_known_frame.grid(row=0, column=2, columnspan=15, rowspan=3, padx=5)
    selected_data_label.grid(sticky=W, row=0, column=1)
    target_variable_label.grid(sticky=W, row=2, column=0)
    target_variable_combo_box.grid(sticky=W, row=2, column=1)
    data_known_label.grid_forget()
    data_known_label.grid(sticky=W, row=0, column=2, columnspan=15, padx=11)
    csv_viewer_button.grid(sticky=W, row=4, column=0)





def app_cleaner():
    target_variable = target_variable_combo_box.get()

    # Grabs the inputs from the features known and values known boxes
    combo_box_entries_list = []
    for combo_box_entry in features_we_know_list:
        combo_box_entries_list.append(str(combo_box_entry.get()))
    print('combo box entries list:', combo_box_entries_list)

    values_we_know_entries = []
    for values in values_we_know_list:
        values_we_know_entries.append(str(values.get()))
    print('values we know entries:', values_we_know_entries)


    # put all items from all entries list into a dictionary value
    global data_we_know_dict
    data_we_know_dict = {}
    runner = 0
    for value in values_we_know_entries:
        data_we_know_dict[combo_box_entries_list[runner]] = [value]
        runner += 1
    del data_we_know_dict['']
    print('Data We Know Dictionary:', data_we_know_dict)

    # Checks if the selected target variable is the right type
    if original_df[target_variable].dtypes == 'object' or original_df[target_variable].dtypes == 'bool':
        target_variable_error_label = Label(model_runner_frame, text='ERROR: TARGET VARIABLE'
        ' MUST BE NUMERIC', fg='red', bg='gray17', font=('Helvetica', 9))
        target_variable_error_label.grid(sticky=W, row=7, column=0)

    else:
        target_variable_error_label = Label(text='')
        from Step_1_Visualizing.visualization import data_type_cleaner
        data_type_cleaner(original_df, target_variable)

        type_clean_df = visualization.data_type_cleaner(original_df, target_variable)
        from Step_2_Single_Encoding.single_hot_encoder import single_encoder
        single_encoded_df = single_encoder(type_clean_df)

        from Step_3_Multiple_Encoding.multiple_hot_encoder import multiple_encoder
        global multiple_encoded_df
        multiple_encoded_df = multiple_encoder(original_df, single_encoded_df)

    # Widgets
    scale_button = Button(model_runner_frame, text='Scale', bg='orange', command=app_scaler)
    global scaler_runtimes_entry
    scaler_runtimes_entry = Entry(model_runner_frame, width=20)
    scaler_runtimes_entry.insert(0, 'Scaler Runtimes')

    # Locations
    scaler_runtimes_entry.grid(sticky=W, row=10, column=0)
    scale_button.grid(sticky=W, row=11, column=0)

def app_visualizer():
    global target_variable
    target_variable = target_variable_combo_box.get()

    # Stops the wrong target variables from being accepted
    if original_df[target_variable].dtypes == 'object' or original_df[target_variable].dtypes == 'bool':
        target_variable_error_label = Label(model_runner_frame, text='ERROR: TARGET VARIABLE'
        ' IS NON-NUMERIC', fg='red', bg='gray17', font=('Helvetica', 9))
        target_variable_error_label.grid(sticky=W, row=6, column=0)

    else:
        global type_clean_df
        type_clean_df = visualization.data_type_cleaner(original_df, target_variable)
        visualization.main_visualizer(type_clean_df, target_variable)

def app_scaler():
    target_variable = target_variable_combo_box.get()
    scaler_runtimes = int(scaler_runtimes_entry.get())
    from Step_5_Scaling.scaler import main_scaler
    global scaled_df, scaled_data_we_know_df
    scaled_df, scaled_data_we_know_df = main_scaler(scaler_runtimes, multiple_encoded_df, target_variable, data_we_know_dict)

    # Widgets
    global importance_finder_runtimes_entry
    importance_finder_runtimes_entry = Entry(model_runner_frame, width=30)
    importance_finder_runtimes_entry.insert(0, 'Importance Finder Runtimes')
    new_importance_finder_button = Button(model_runner_frame, text='Find New Important Features', bg='cyan', command=app_new_important_feature_finder)
    important_features_wanted_combo = ttk.Combobox(model_runner_frame, value=[x + 1 for x in range(100)], width=30)
    important_features_wanted_label = Label(model_runner_frame, text='Select # of Features Wanted')
    saved_important_features_button = Button(model_runner_frame, text='Use Saved Important Features', bg='purple', command=app_saved_important_features_opener)

    # Locations
    importance_finder_runtimes_entry.grid(sticky=W, row=12, column=0)
    new_importance_finder_button.grid(sticky=W, row=13, column=0)
    important_features_wanted_combo.grid(sticky=W, row=12, column=1)
    important_features_wanted_label.grid(sticky=W, row=11, column=1)
    saved_important_features_button.grid(sticky=W, row=13, column=1)


def app_saved_important_features_opener():
    important_features_window = Tk()
    important_features_window.title('Select Features You Want')
    important_features_window.geometry('900x300')

    # Create the tree
    tree_important_features_window = ttk.Treeview(important_features_window)

    # Define the tree's columns
    tree_important_features_window['columns'] = ('Dataframe', 'Most Important Features', 'Runtimes')

    # Format the tree's columns
    tree_important_features_window.column('#0', width=0, stretch=NO)
    tree_important_features_window.column('Dataframe', anchor=W, width=140)
    tree_important_features_window.column('Most Important Features', width=620)
    tree_important_features_window.column('Runtimes', width=120)

    # Create tree's headings
    tree_important_features_window.heading('Dataframe', text='Dataframe', anchor=W)
    tree_important_features_window.heading('Most Important Features', text='Most Important Features')
    tree_important_features_window.heading('Runtimes', text='Runtimes')

    # Add data
    data = [['nfl_data.csv', ['total_yards', 'Humidity'], '100'
             ]]
    # Insert data
    global count
    count = 0
    for record in data:
        print('Record:', record)
        tree_important_features_window.insert(parent='', index='end', iid=count, text='',
                                              values=(record[0], record[1], record[2]))
        count += 1

    def tree_use_selected_features():
        # ToDo push the selected tree to the model runner
        important_features_window.destroy()

    def tree_clicked_features(event):
        tree_use_selected_features()

    def tree_cancellation():
        important_features_window.destroy()

    def tree_remove_selected():
        selected = tree_important_features_window.selection()[0]
        tree_important_features_window.delete(selected)

    # Widgets
    tree_select_features_button = Button(important_features_window, text='Use Selected Features',
                                         command=tree_use_selected_features)
    tree_cancel_button = Button(important_features_window, text='Cancel', command=tree_cancellation)
    tree_remove_button = Button(important_features_window, text='Remove Selected', command=tree_remove_selected)
    tree_clicker_binding = tree_important_features_window.bind('<Double-1>', tree_clicked_features)

    # Locations
    tree_important_features_window.grid(row=0, column=0, columnspan=21)
    tree_select_features_button.grid(row=1, column=9)
    tree_cancel_button.grid(row=1, column=11)
    tree_remove_button.grid(row=1, column=10)

    important_features_window.mainloop()


def app_new_important_feature_finder():
    target_variable = target_variable_combo_box.get()
    importance_finder_runtimes = int(importance_finder_runtimes_entry.get())
    from Step_6_Feature_Importance_Finding.importance_finder import feature_importer_non_printing
    feature_importer_non_printing(importance_finder_runtimes, scaled_df, target_variable)

    # Buttons
    feature_combiner_button = Button(model_runner_frame, text='Run Feature Combiner', bg='pink', command=app_feature_combiner)

    # Locations
    feature_combiner_button.grid(sticky=W, row=14, column=0)
    # ToDo add in combo box for amount of features wanted

def app_feature_combiner():
    target_variable = target_variable_combo_box.get()
    from Step_7_Feature_Combination_Testing.feature_selection import feature_combiner
    feature_combiner(target_variable)
    pass


def target_variable_combo_click(event):
    clicked_target_variable = target_variable_combo_box.get()
    print('Clicked Target Variable:', clicked_target_variable)
    if original_df[clicked_target_variable].dtypes == 'object' or original_df[clicked_target_variable].dtypes == 'bool':
        target_variable_error_label = Label(model_runner_frame, text='ERROR: TARGET VARIABLE'
        ' MUST BE NUMERIC', fg='red', bg='gray17', font=('Helvetica', 9))
        target_variable_error_label.grid(sticky=W, row=7, column=0)
        visualization_button.config(state=DISABLED)
        clean_button.config(state=DISABLED)

    else:
        visualization_button.config(state=NORMAL)
        clean_button.config(state=NORMAL)
        target_variable_error_label = Label(model_runner_frame, text='Acceptable Selection', bg='gray17', fg='green')
        target_variable_error_label.grid(sticky=W, row=7, column=0, ipadx=100)

# data
model_options = [
            'Linear Regression',
            'Logistic Regression',
            'KNN',
            'SVD']

clicked_model = StringVar()
clicked_target_variable = StringVar()
# frames
visualization_frame = Frame(model_runner_frame)


# ToDo add in scrollbar to model runner page

# labels
model_chooser_label = Label(model_runner_frame, text='Select Model', font=('Helvetica', 10), width=16)
target_variable_label = Label(model_runner_frame, text='Select Target Variable', font=('Helvetica', 10))
# visualization_temporary_label = Label(visualization_frame, text='Visualization', background='blue')

# widgets
csv_opener_button = Button(model_runner_frame, text='Select Data', command=app_csv_opener, width=18, height=1)
model_combo_box = ttk.Combobox(model_runner_frame, value=model_options, width=21)
model_combo_box.current(0) # sets the initial box focus
clean_button = Button(model_runner_frame, text='Clean', bg='pink', command=app_cleaner)
visualization_button = Button(model_runner_frame, text='Visualize', bg='yellow', command=app_visualizer)


# locations
# visualization_frame.grid(sticky=W, row=0, column=2)
# visualization_temporary_label.grid(row=0, column=0, ipadx=600, ipady=20, rowspan=15)
csv_opener_button.grid(sticky=W, row=0, column=0)
model_chooser_label.grid(sticky=W, row=1, column=0)
model_combo_box.grid(sticky=W, row=1, column=1)
visualization_button.grid(sticky=W, row=8, column=1, pady=5)
clean_button.grid(sticky=W, row=8, column=0, pady=5)

root.mainloop()


