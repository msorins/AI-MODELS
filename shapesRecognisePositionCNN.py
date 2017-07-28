#https://github.com/ZFTurbo/KAGGLE_DISTRACTED_DRIVER/blob/master/run_keras_simple.py

#Part 1 -> Importing
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



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set_X = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = None)

testing_set_X = test_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = None)


#Part 5 -> Get the CSV files

n_training = int(training_set_X.n / 2)
training_set_y = []

crt = 0

for i in range(n_training):
    data = genfromtxt(os.getcwd() + '/datainfo/training_set/circle/' + str(crt) + '.csv', delimiter=',')
    training_set_y.append(numpy.array(data))
    crt += 1

for i in range(n_training):
    data = genfromtxt(os.getcwd() + '/datainfo/training_set/square/' + str(crt) + '.csv', delimiter=',')
    training_set_y.append(numpy.array(data))
    crt += 1


n_testing = int(testing_set_X.n / 2)
testing_set_y = []

for i in range(n_testing):
    data = genfromtxt(os.getcwd() + '/datainfo/test_set/circle/' + str(crt) + '.csv', delimiter=',')
    testing_set_y.append(numpy.array(data))
    crt += 1

for i in range(n_testing):
    data = genfromtxt(os.getcwd() + '/datainfo/test_set/square/' + str(crt) + '.csv', delimiter=',')
    testing_set_y.append(numpy.array(data))
    crt += 1

training_set_y = numpy.array(training_set_y)
testing_set_y = numpy.array(testing_set_y)

#Part 5 -> fit the  NN
classifier.fit(training_set_X,
               training_set_y,
               batch_size=32,
               epochs=50,
               validation_data=(testing_set_X, testing_set_y),
               verbose=2)
