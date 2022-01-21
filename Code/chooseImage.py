# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt
 
Tk().withdraw()
filename = askopenfilename()
print(filename)
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
image = cv2.imread(filename)
oriImage = image.copy()
 
def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping
 
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2:
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
            
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]


            sample = roi
            white_image = np.zeros(sample.shape) + 255
            I = cv2.resize(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), (700,600))

            #denoise
            I=cv2.fastNlMeansDenoising(I)
            I=cv2.fastNlMeansDenoising(I)
            I=cv2.fastNlMeansDenoising(I) 
            B = cv2.adaptiveThreshold(I,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            cv2.imshow('B', B)
            median = B

            for i in range(0,7):
                median = cv2.medianBlur(median,3)
                
            ctr, hierarchy = cv2.findContours(B, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ctr = list(filter(lambda el : el.shape[0]>5 and el.shape[0]<200, ctr))

            for i in range(0,100):
                cv2.drawContours(white_image, ctr, 1, (0,0,255))

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
            cv2.imshow("filtered", I_2)   

            i=0
            resized = []
            filtered.sort(key = lambda x: x[0])
            for f in filtered:

                t = cv2.bitwise_not(median[f[1] - int(f[3]*0.1):f[1]+int(f[3]*1.1), f[0] - int(f[2]*0.1):f[0]+int(1.1*f[2])])
                t2 = cv2.subtract(255, t) 
                resized = cv2.resize(t2, (45,45))
                cv2.imwrite("test//0000" + str(i) + ".jpg", resized)
                i+=1

 
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

while (cv2.waitKey(1)!=27):
 
    i = image.copy()
 
    if not cropping:
        cv2.imshow("image", image)

 
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)

    cv2.waitKey(1)
cv2.destroyAllWindows()

os.system('python classification.py')
