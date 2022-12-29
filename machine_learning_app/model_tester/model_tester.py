from tkinter import *
from tkinter import messagebox

root = Tk()
root.title('Model Tester')
root.geometry('400x400')

chosen_model = IntVar() # intvar is always used for checkboxes, but use StringVar to return something other than an int
chosen_model.set('')

models = []
chosen_models = []

my_label = Label(root, text='Choose Which Models to Test').pack()
bottom_label = Label(text='')

def checkbox_grabber():
    for i in range(len(models)):
        selected = ''
        if models[i].get()>=1:
            selected += str(i)
            messagebox.showinfo(message="You selected Checkbox " + selected)
        chosen_models.append(selected)
    print(chosen_models)



for i in range(5): # choose how many checkboxes you will have plus 1
    chosen_model = IntVar()
    chosen_model.set(0)
    models.append(chosen_model)



# the onvalue changes what the selected box will return. Usually it will return a 1 if selected and a 0 if not selected
Checkbutton(root, text='Linear Regression', variable=models[1]).pack()

Checkbutton(root, text='Logistic Regression', variable=models[2]).pack()

Checkbutton(root, text='KNN', variable=models[3]).pack()

Checkbutton(root, text='SVD', variable=models[4]).pack()

submit_button = Button(root, text='Submit', command=checkbox_grabber).pack()


root.mainloop()