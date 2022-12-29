from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

root = Tk()
img = Image.open('sort_down_arrow.png')
resizer = img.resize((20,20), Image.Resampling.LANCZOS)
resized_img = ImageTk.PhotoImage(resizer)



# tree = ttk.Treeview( root , column=("c1","c2","c3") , height = 10 )



tree = ttk.Treeview(root)

tree['columns'] = ('column1', 'column2')
tree.column('column1', width=80, stretch=NO)
tree.column('column2', minwidth=800, stretch=NO)
tree.heading('column1', text='icon', anchor=W, image=resized_img)
tree.pack(expand=True, fill=BOTH)

root.mainloop()