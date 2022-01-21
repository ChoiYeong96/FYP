from tkinter import *
from tkinter import ttk
import os
root = Tk()

def start(arg):
    if arg == 1:
        os.system('python chooseImage.py')
    elif arg == 2:
        os.system('python drawExpression.py')
        
        
b1 = Button(root, text="Choose Image", command=lambda: start(1))
b1.pack()
b2 = Button(root, text="Draw an Expression", command=lambda: start(2))
b2.pack()
