from tkinter import *
from tkinter import ttk

root = Tk()
root.title('Model Runner')
root.geometry('400x400')


model_chooser_label = Label(root, text='Choose Your Model').pack()

def show():
    label = Label(root, text=model_combo.get()).pack()
    print(model_combo.get())

models = ['Linear Regression', 'Logistic Regression', 'KNN', 'Decision Tree', 'SVM', 'K-Means', 'Naive Bayes', 'Random Forest']

model_combo = ttk.Combobox(root, value=models)
model_combo.current(0)
model_combo.pack()


button = Button(root, text='Show Selection', command=show).pack()












mainloop()