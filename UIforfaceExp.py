import tkinter as tk
from tkinter import Message, Text
import tkinter.ttk as ttk
import tkinter.font as font
import os
# import shutil
# import csv                    # to be added when working on the Expression_Recognition model
# import numpy as np
# from PIL import Image, ImageTk
# import pandas as pd
# import cv2
#from pathlib import Path
# import datetime
# import time

 
window = tk.Tk()
window.title("Expression_Recognition")
window.configure(background ='Orange')
window.grid_rowconfigure(0, weight = 2)
window.grid_columnconfigure(0, weight = 2)
message = tk.Label(
    window, text ="Expression-Recognition-System",
    bg ="Yellow", fg = "Red", width = 50,
    height = 3, font = ('times', 30, 'bold'))
     
message.place(x = 200, y = 20)
 
box= tk.Label(window, text = "User Sr. No",
width = 20, height = 2, fg ="Red",
bg = "Yellow", font = ('times', 15, ' bold ') )
box.place(x = 400, y = 200)
 
userinp = tk.Entry(window,   # Takes the Sr.No of the users coming on the platform 
width = 20, bg ="yellow",
fg ="Red", font = ('times', 15, ' bold '))
userinp.place(x = 700, y = 215)
 
box1 = tk.Label(window, text ="Please add your Name",
width = 20, fg ="Red", bg ="Yellow",
height = 2, font =('times', 15, ' bold '))
box1.place(x = 400, y = 300)
 
userinp1 = tk.Entry(window, width = 20, #Takes the Name of the user who wants to check the Expression
bg ="Yellow", fg ="Red",
font = ('times', 15, ' bold ')  )
userinp1.place(x = 700, y = 315)

window.mainloop()

  
