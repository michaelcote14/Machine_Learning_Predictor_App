from tkinter import *
import pandas as pd
from tkinter import ttk, filedialog


csv_viewer_window = Tk()
csv_viewer_window.title('CSV Tree View')
csv_viewer_window.geometry('1200x300')

# Create frame
csv_viewer_frame = Frame(csv_viewer_window)
csv_viewer_frame.pack()

# Create treeview
csv_viewer_tree = ttk.Treeview(csv_viewer_frame)

def file_open():
    csv_file_location = filedialog.askopenfilename(initialdir='/', title='Open A CSV File',
                filetype=(('CSV Files', '*.csv'), ('All Files', '*.*')))

    if csv_file_location:
        try:
            csv_file_name = r'{}'.format(csv_file_location)
            df = pd.read_csv(csv_file_name)
            print('original df', df)
        except ValueError:
            file_opener_error_label.config(text= 'File could not be opened. Try again.')
        except FileNotFoundError:
            file_opener_error_label.config(text='File could not be found')

    # Clears the current tree
    clear_tree()

    # Set up new treeview
    csv_viewer_tree['column'] = list(df.columns)
    csv_viewer_tree['show'] = 'headings'

    # Loop through column list to create the tree headers
    for column in csv_viewer_tree['column']:
        csv_viewer_tree.heading(column, text=column)

    # Put data in treeview
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        csv_viewer_tree.insert('', 'end', values=row)

    csv_viewer_tree.pack()

def clear_tree():
    # Deletes all the records in the old tree
    csv_viewer_tree.delete(*csv_viewer_tree.get_children())

# Add a menu
my_menu = Menu(csv_viewer_window)
csv_viewer_window.config(menu=my_menu)

# Add menu dropdown
file_menu = Menu(my_menu, tearoff=False) # tearoff takes the small dots away
my_menu.add_cascade(label='Spreadsheets', menu=file_menu)
file_menu.add_command(label='Open', command=file_open)

file_opener_error_label = Label(csv_viewer_window, text='')
file_opener_error_label.pack(pady=20)


csv_viewer_window.mainloop()
