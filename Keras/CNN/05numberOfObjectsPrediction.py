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
from keras.preprocessing.sequence import pad_sequences
from numpy import genfromtxt
from keras.utils import np_utils
import numpy
import os
import cv2
import sys

from keras import backend as K
K.set_image_dim_ordering('th')

cwd = os.getcwd()

#Part 2 -> Initialise the CNN
img_rows, img_cols = 256, 256
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))

classifier.add(Activation('relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 5 -> Flattening
classifier.add(Flatten())

# Step 6 -> Full connection
classifier.add(Dense(output_dim = 1024, activation = 'relu'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 1024, activation = 'relu'))
#classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 2048, activation = 'relu'))
classifier.add(Dense(output_dim = 1))

#Part 3 -> Compile the CNN
classifier.compile(optimizer = 'adadelta', loss = 'mse', metrics = ['accuracy'])

#Part 4 -> Get the images & csv
def get_im(path):
    # Load as grayscale
    print("Path: ", path)
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

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

X_train = numpy.array(X_train, dtype = numpy.uint8)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_train /= 255


X_test = numpy.array(X_test, dtype = numpy.uint8)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_test = X_test.astype('float32')
X_test /= 255

#Part 5 -> fit the  NN
classifier.fit(numpy.array(X_train),
               numpy.array(y_train),
               batch_size=64,
               nb_epoch=500,
               validation_data=(
                   numpy.array(X_test),
                   numpy.array(y_test)
               ),
               verbose=2,
               )

#Saving path is
pth = str( os.getcwd() + "/" + sys.argv[0] + ".h5")
print("Path is: ", pth)
classifier.save(filepath = pth)
