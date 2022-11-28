from tkinter import *
from tkinter import ttk

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
data = [['nfl_data.csv', ['total_yards', 'Humidity'], '100'],
        ['nfl_data(shortened).csv', ['total_yards', 'pos'], '1000']
        ]
# Insert data
global count
count = 0
for record in data:
    print('Record:', record)
    tree_important_features_window.insert(parent='', index='end', iid=count, text='', values=(record[0], record[1], record[2]))
    count += 1

def tree_feature_selector():
    important_features_window.destroy()


def tree_cancellation():
    important_features_window.destroy()

def remove_all():
    for record in tree_important_features_window.get_children():
        tree_important_features_window.delete(record)

def remove_many():
    selected = tree_important_features_window.selection()
    for record in selected:
        tree_important_features_window.delete(record)

def remove_one():
    selected = tree_important_features_window.selection()[0] # gets the unique id of the record
    tree_important_features_window.delete(selected)


def add_record():
    tree_important_features_window.insert(parent='', index='end', iid=5, values=(add_record_entry1.get(), add_record_entry2.get(), add_record_entry3.get()))

def printer(event):
    # Grab the selected record number
    selected = tree_important_features_window.focus()
    # Grab the record values
    values = tree_important_features_window.item(selected, 'values')
    print('Values:', values)


# Create frame for buttons
button_frame = Frame(important_features_window)

# Widgets
tree_select_features_button = Button(button_frame, text='Choose Features', command=tree_feature_selector)
tree_cancel_button = Button(button_frame, text='Cancel', command=tree_cancellation)
remove_all_button = Button(button_frame, text='Remove All Records', command=remove_all)
add_record_button = Button(button_frame, text='Add Record', command=add_record)
add_record_entry1 = Entry(button_frame)
add_record_entry2 = Entry(button_frame)
add_record_entry3 = Entry(button_frame)
remove_one_button = Button(button_frame, text='Remove One', command=remove_one)
remove_many_button = Button(button_frame, text='Remove Many', command=remove_many)
tree_important_features_window.bind('<Double-1>', printer)

# Locations
tree_important_features_window.grid(row=0, column=0, columnspan=21)
button_frame.grid(row=1, column=0, columnspan=21)
tree_select_features_button.grid(row=1, column=1)
tree_cancel_button.grid(row=1, column=5, padx=5)
remove_all_button.grid(row=1, column=4, padx=5)
add_record_button.grid(row=3, column=3, padx=5)
add_record_entry1.grid(row=3, column=0)
add_record_entry2.grid(row=3, column=1)
add_record_entry3.grid(row=3, column=2)
remove_one_button.grid(row=1, column=2, padx=5)
remove_many_button.grid(row=1, column=3, padx=5)



important_features_window.mainloop()