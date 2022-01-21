# For AVX and SSE instructions
import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, ZeroPadding2D, Activation
from keras.layers.normalization import BatchNormalization
import sys
import h5py

print("Converting to Latex...")

# Image attributes
img_width, img_height = 45, 45
num_classes = 82
CHANNELS = 1

#list of class(symbol) names
path ='dataset'
nameList = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'alpha', 'ascii_124', 'b', 'beta', 'C', 'cos', 'd', 'Delta', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'G', 'gamma', 'geq', 'gt', 'H', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'M', 'mu', 'N', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'R', 'rightarrow', 'S', 'sigma', 'sin', 'sqrt', 'sum', 'T', 'tan', 'theta', 'times', 'u', 'v', 'w', 'X', 'y', 'z', '[', ']', '{', '}']
    
# Function to create model
def create_model():
    model = Sequential()

    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(img_height, img_width, CHANNELS)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5), padding="same", input_shape=(img_height, img_width, CHANNELS)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    model.add(Dense(num_classes, activation='softmax'))

    return model

test_folder = 'test'
list3 = []
c = 1
for i in os.listdir(test_folder):
    # create a grid of 3x3 images
    plt.subplot(7, 7, c)
    c += 1
    image = cv2.imread(test_folder + "\\" + i, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (45, 45))
    image = np.array(image).reshape((img_width, img_height, 1))
    plt.imshow(image.reshape(45, 45), cmap=plt.get_cmap('gray'))

    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)

    # Creating and loading model
    model = create_model()
    
    # Load trained nueral nets
    model.load_weights('./output/output.h5py')
    hf = h5py.File('output/output.h5py', 'r')
    print(hf.keys())

    prediction = model.predict(image)[0]

    def percent(num1, num2):
        num1 = float(num1)
        num2 = float(num2)
        percentage = '{0:.2f}'.format((num1 / num2 * 100))
        print("percentage",percentage)
        return percentage

    pred_dict = {}
    i = 0
    for listitem in list(prediction):
        pred_dict[i] = listitem
        i += 1
    
    for w in sorted(pred_dict, key=pred_dict.get, reverse=True)[:1]:
        prediction_label = nameList[w]
        print("symbol name:",prediction_label)
        prediction_conf =  percent(pred_dict[w], 1)
        list3.append(prediction_label)
        
        
    
    plt.tight_layout()
    plt.xlabel(prediction_label)

# # show the plot
#plt.show()

def transform_to_latex(handwritten_text):
    latex_symbols_dict = {
        '-' : '-',',' : ',','!' : '!','(' : '(',')' : ')',
        '[' : '[',']' : ']','{' : '{','}' : '}','+' : '+','=' : '=',
        '0' : '0','1' : '1','2' : '2','3' : '3','4' : '4','5' : '5',
        '6' : '6','7' : '7','8' : '8','9' : '9','A' : 'A','alpha' : '\alpha',
        'ascii_124' : '|','b' : 'b','beta' : '\beta','C' : 'C','cos' : '\cos','d' : 'd',
        'Delta' : '\delta','div' : '\div','e' : 'e','exists' : '\exists','f' : 'f','\forall' : 'forall',
        'forward_slash' : '/','G' : 'G','gamma' : '\gamma','geq' : '\geq','gt' : '>','H' : 'H',
        'i' : 'i','in' : '\in','infty' : '\infty','int' : '\int','j' : 'j','k' : 'k',
        'l' : 'l','lambda' : '\lambda','ldots' : '\ldots','leq' : '\leq','\lim' : 'lim','log' : '\log',
        'lt':'<','M':'M','mu':'\mu','N':'N','neq':'\neq','o':'o','p':'p','phi':'\phi','pi':'\pi',
        'pm':'\pm','prime':'\prime','q':'q','R':'R','rightarrow':'\Rightarrow','S':'S','sigma':'\sum',
        'sin':'\sin','sqrt':'\sqrt','sum':'\sum','T':'T','tan':'\tan','theta':'\theta','times':'\times',
        'u':'u','v':'v','w':'w','X':'X','y':'y','z':'z'
    }
    return latex_symbols_dict.get(handwritten_text)
list2 = []

for i in list3:
    a = transform_to_latex(i)
    print(a)
    list2.append(a)

print(list2)
a = " ".join(list2)
ax = plt.axes([0,0,1,1.2]) #left,bottom,width,height
ax.set_xticks([])
ax.set_yticks([])
plt.text(0.2,0.4,'$%s$' %a,size=40)
plt.show()
plt.close()
os.chdir(test_folder)
files = glob.glob('*.jpg') 
for file in files:
    os.unlink(file)
