from tkinter import *
import PIL
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os

def submit():
    
    global image_number 
    filename = f'testDraw.png'   
    image1.save(filename)
    image_number += 1
    def checkIfContains(outside, inside):        
        return (inside[0] > outside[0] and inside[0] + inside[2]  < outside[0] + outside[2] and inside[1] > outside[1] and inside[1] + inside[3] < outside[1] + outside[3])

    def filterRectangles(rects):
        rects_filtered = []
        flag = 0
        for i in range(0, len(rects)):
            for j in range(0, len(rects)):
                if(checkIfContains(rects[j], rects[i])):
                    flag = 1
                    break
            if(flag == 0):
                rects_filtered.append(rects[i])
            else:
                flag = 0
        return rects_filtered


    sample = cv2.imread('testDraw.png')
    white_image = np.zeros(sample.shape) + 255
    I = cv2.resize(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), (700,600))
    #cv2.imshow("sample", I)
          
    B = cv2.adaptiveThreshold(I,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.imshow('B', B)
    median = B

    for i in range(0,7):
        median = cv2.medianBlur(median,3)
        
    ctr, hierarchy = cv2.findContours(B, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr = list(filter(lambda el : el.shape[0]>5 and el.shape[0]<200, ctr))

    for i in range(0,100):
        cv2.drawContours(white_image, ctr, -1, (0,0,255))

    rects = []
    i=0
    I_1 = B.copy()
    for c in ctr:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        rect = (x, y, w, h) 
        rects.append(rect)

    filtered = filterRectangles(rects)

    I_2 = B.copy()
    for i in range(0, len(filtered)):
        cv2.rectangle(I_2, (filtered[i][0], filtered[i][1]), (filtered[i][0]+filtered[i][2], filtered[i][1]+filtered[i][3]), (0, 255, 0), 1);
    #cv2.imshow("segmented", I_2)   

    i=0
    resized = []
    filtered.sort(key = lambda x: x[0])
    for f in filtered:

        t = cv2.bitwise_not(median[f[1] - int(f[3]*0.1):f[1]+int(f[3]*1.1), f[0] - int(f[2]*0.1):f[0]+int(1.1*f[2])])
        t2 = cv2.subtract(255, t)
        resized = cv2.resize(t2, (45,45))
        cv2.imwrite("test//0000" + str(i) + ".jpg", resized)
        i+=1
    os.system('python classification.py')
    
def clear():
    cv.delete('all')
    filename = f'testDraw.png'
    if os.path.exists(filename):
        os.remove(filename)
        

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=4)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=4)
    lastx, lasty = x, y


root2 = Tk()

lastx, lasty = None, None
image_number = 0

cv = Canvas(root2, width=640, height=480, bg='white')
# --- PIL
image1 = PIL.Image.new('RGB', (640, 480), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_submit = Button(text="submit", command=submit)
btn_submit.pack(side = "right")
btn_clear = Button(text="clear", command=clear)
btn_clear.pack(side = "right")
root2.mainloop()

