#https://github.com/ZFTurbo/KAGGLE_DISTRACTED_DRIVER/blob/master/run_keras_simple.py

#Part 1 -> Importing
import glob

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from numpy import genfromtxt
import numpy
import os
import cv2

#Part 2 -> Initialise the CNN

classifier = Sequential()

# Step 1 -> Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 -> Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 -> Convolution
classifier.add(Convolution2D(32, 3, 3,  activation = 'relu'))

# Step 4 -> Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 5 -> Flattening
classifier.add(Flatten())

# Step 6 -> Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 4))

#Part 3 -> Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 4 -> Get the images & csv
def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (128, 96))
    return resized

def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in ["circle", "square"]:
        print('Load folder c{}'.format(j))
        path = os.path.join('dataset', 'training_set', j, '*.png')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)

            flCSV = fl.split('.')
            flCSV[-1] = "csv"
            flCSV = ".".join(flCSV)

            y_train.append( genfromtxt(flCSV) )

    return X_train, y_train


def load_test():
    X_test = []
    y_test = []
    print('Read test images')
    for j in ["circle", "square"]:
        print('Load folder c{}'.format(j))
        path = os.path.join('dataset', 'test_set', j, '*.png')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_test.append(img)

            flCSV = fl.split('.')
            flCSV[-1] = "csv"
            flCSV = ".".join(flCSV)

            y_test.append(genfromtxt(flCSV))

    return X_test, y_test



X_train, y_train = load_train()
X_test, y_test = load_test()

#Part 5 -> fit the  NN
classifier.fit(X_train,
               y_train,
               batch_size=32,
               epochs=50,
               validation_data=(X_test, y_test),
               verbose=2)
