import os, glob
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time
import os.path as path
import keras
import keras.backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Activation
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import pickle
import numpy as np
from imutils import paths
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

np.random.seed(2017)

MODEL_NAME = 'convneted model'
EPOCHS = 25
BATCH_SIZE = 16
CHANNELS = 1
num_classes = 82
img_height = 45
img_width = 45

def load_data(datasetPath):

    # load data from the pickle file
    with open(datasetPath, 'rb') as f:
        data, labels = pickle.load(f)

    # partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
    print("\n splitting dataset into train and validation sets")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    print("\n reshaping images")
    x_train = x_train.reshape(x_train.shape[0], img_height, img_width, CHANNELS)
    x_test = x_test.reshape(x_test.shape[0], img_height, img_width, CHANNELS)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def build_model():
    print("\n creating model")
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

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

def train(model, x_train, y_train, x_test, y_test):

    print("\n model training starting\n")
    model.compile(loss=keras.losses.categorical_crossentropy, \
                  optimizer=keras.optimizers.Adam(), \
                  metrics=['accuracy'])

    # define data preparation
    datagen = ImageDataGenerator(
                # featurewise_center=True,
                # featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2)
                # horizontal_flip=True,
                # vertical_flip=True)

    # fit parameters from data
    datagen.fit(x_train)

    # start timer
    start = time.time()

    # fits the model on batches with real-time data augmentation:
    model_info = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=EPOCHS, 
                        validation_data=(x_test, y_test))
    
    # end timer and print the time taken to train model
    end = time.time()
    print("\nModel took %0.2f seconds to train"%(end - start))

    # display graph accuracy and loss
    plot_model_history(model_info)

    # compute test accuracy
    print("Accuracy on test data is: %0.2f"%accuracy(x_test, y_test, model))


def export_model(saver, model, input_node_names, output_node_name):
    # Save trained model in .h5py file
    print("\n exporting model to .h5py file")
    hf = h5py.File('output/output', 'r')
    print(hf.keys())
    model.save_weights('output/output.h5py')

    print("\n exporting model to .pb file\n")
    
    print("\n model saved")

def main():

    print("\n checking for 'output' folder if exists")
    if not path.exists('output'):
        print("\n output folder not available, creating folder 'output' in the root directory")
        os.mkdir('output')

    print("\n loading data from the dataset into variables")
    x_train, y_train, x_test, y_test = load_data('dataset_pickle.pickle')

    # train model
    model = build_model()
    # model.load_weights('./output-checkpoint/output.h5py')
    train(model, x_train, y_train, x_test, y_test)

    # export model
    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")

if __name__ == '__main__':
    main()
