from keras.models import load_model
from tkinter import *
import tkinter as tk
from win32 import win32gui
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image
import numpy as np
import joblib

import cv2
import os

model = joblib.load('digit_model1.joblib')
b=0
def predict_digit(img):
    global b
    b=b+1
    #resize image to 28x28 pixels
    img = img.resize((28,28),Image.ANTIALIAS)
    #convert rgb to grayscale
    img = img.convert('L')
    plt.imshow(img,cmap='gray')
    #plt.show()
    #plt.imsave('a3.jpg',img)
    #cv2.imwrite('a4.jpg',img)
    img = np.array(img)
    #image=Image.fromarray(img)
    #image.save('a5.jpg')
    a=cv2.subtract(255,img)
    cv2.imwrite('d'+str(b)+'.jpg',a)
    #reshaping to support our model input and normalizing
    img = a.reshape((1,-1))
    print(img)
    #img=255-img
    #print(img)
    #predicting the class
    res = model.predict(img)
    
    return res

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=224, height=224, bg = "white", cursor="spraycan")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text = "Recognise", command =self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure

        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        a,b,c,d=rect
        rect=(a+4,b+4,c-4,d-4)
        im = ImageGrab.grab(bbox=rect)
        #img = Image.open('9.jpg')
        digit = predict_digit(im)
        self.label.configure(text= str(digit))
        #+', '+ str(int(acc*100))+'%'

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()
