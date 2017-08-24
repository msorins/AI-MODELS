#https://github.com/ZFTurbo/KAGGLE_DISTRACTED_DRIVER/blob/master/run_keras_simple.py

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3" # Will use only 2nd, 3rd and 4th gpu


#Part 1 -> Importing
import glob

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
from numpy import genfromtxt
from keras.utils import np_utils
import tensorflow
import numpy
import os
import cv2
import sys

from keras import backend as K
K.set_image_dim_ordering('th')

cwd = os.getcwd()

# Part 2 -> Initialise the CNN
img_rows, img_cols = 256, 256
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

def build_classifier():
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=(1, img_rows, img_cols)))

    classifier.add(Activation('relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Dropout(0.2))

    # Adding a second convolutional layer
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))


    # Step 5 -> Flattening
    classifier.add(Flatten())

    # Step 6 -> Full connection
    classifier.add(Dense(output_dim = 1024, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim = 1024, activation = 'relu'))
    classifier.add(Dense(output_dim = 2048, activation = 'relu'))
    classifier.add(Dense(output_dim = 1))

    #Part 3 -> Compile the CNN
    classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])


    return classifier

#Part 4 -> Get the images & csv
def get_im(path):
    # Load as grayscale
    print("Path: ", path)
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized
""""
def load_train():
    y_train = []

    path = os.path.join('datasetv2', 'training_set', '*.csv')
    files = glob.glob(path)

    for fl in files:
         text = str(genfromtxt(fl, deletechars=",", dtype=None))
         text = text.split(",")
         y_train.append(len(text) / 4)
         print(str(len(text) / 4) + " images")

    return y_train


def load_test():
    y_test = []

    path = os.path.join('datasetv2', 'test_set', '*.csv')
    files = glob.glob(path)

    for fl in files:
        text = str(genfromtxt(fl, deletechars=",", dtype=None))
        text = text.split(",")
        y_test.append(len(text) / 4)

        print(str(len(text) / 4) + " images")

    return  y_test
"""

def load_train():
    X_train = []
    y_train = []
    print('Read train images')

    print('Load folder c{}')
    path = os.path.join('datasetv2', 'training_set', '*.png')
    files = glob.glob(path)

    for fl in files:
         if "res" in fl:
            continue

         fl = cwd + "/" + fl
         img = get_im(fl)
         X_train.append(img)

         flCSV = fl.split('.')
         flCSV[-1] = "csv"
         flCSV = ".".join(flCSV)

         text = str(genfromtxt(flCSV, deletechars=",", dtype=None))
         text = text.split(",")
         y_train.append(len(text) / 4)
         print(str(len(text) / 4) + " images")

    return X_train, y_train


def load_test():
    X_test = []
    y_test = []
    print('Read test images')

    print('Load folder c{}')
    path = os.path.join('datasetv2', 'test_set', '*.png')
    files = glob.glob(path)

    for fl in files:
        if "res" in fl:
            continue

        fl = cwd + "/" + fl
        img = get_im(fl)
        X_test.append(img)

        flCSV = fl.split('.')
        flCSV[-1] = "csv"
        flCSV = ".".join(flCSV)

        text = str(genfromtxt(flCSV, deletechars=",", dtype=None))
        text = text.split(",")
        y_test.append(len(text) / 4)
        print(str(len(text) / 4) + " images")

    return X_test, y_test


X_train, y_train = load_train()
X_test, y_test = load_test()

classifier = build_classifier()
#y_train = load_train()
#y_test = load_test()

#Part 4 -> augmentate data
shift = 0.2
targ_size = (256, 256)

#Train data

datagen_train = ImageDataGenerator(
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,)

#Reshape x to have this form (samples, height, width, channels)
X_train = numpy.array(X_train)
X_train = numpy.reshape(X_train, (len(X_train), 1, targ_size[0], targ_size[1]))
X_train = X_train.astype("float64")
#X_train = X_train.reshape((1,) + X_train.shape)
#tensorflow.expand_dims(X_train, 1)
datagen_train.fit(X_train, augment = True)


datagen_test= ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
    )

X_test = numpy.array(X_test)
X_test = numpy.reshape(X_test, (len(X_test), 1, targ_size[0], targ_size[1]))
X_test = X_test.astype("float64")
datagen_train.fit(X_test, augment = True)


datagen_train.standardize(y_train)
datagen_test.standardize(y_test)

#y_train = numpy.array(y_train)
#y_test = numpy.array(y_train)

#Part 5 -> fit the  NN
classifier.fit_generator(datagen_train.flow(X_train, y_train, batch_size= 32),
                         samples_per_epoch = 10000,
                         nb_epoch = 80,
                         validation_data = datagen_test.flow(X_test, y_test),
                         nb_val_samples = 5000
                        )

#Saving path is
pth = str( os.getcwd() + "/" + sys.argv[0] + ".h5")
print("Path is: ", pth)
classifier.save(filepath = pth)
