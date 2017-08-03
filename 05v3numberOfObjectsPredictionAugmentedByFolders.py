import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3" # Will use only 2nd, 3rd and 4th gpu

#Part 1 -> Importing
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#Part 2 -> Initialise the CNN

classifier = Sequential()
classifier.add(
    Convolution2D(32, 3, 3, input_shape=(3, 64, 64), activation='relu')
)
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(256, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dropout(0.2))
classifier.add(Dense(2048, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(512, activation='relu'))
classifier.add(Dense(1024, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 1, activation = 'relu'))

#Part 3 -> Compile the CNN
classifier.compile(optimizer = 'adadelta', loss = 'mse', metrics = ['accuracy'])

#Part 4 -> Get the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datasetv4/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory('datasetv4/test_set',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'sparse',)


#Part 5 -> fit the  NN
"""
classifier.fit_generator(training_set,
                         samples_per_epoch = 2880,
                         nb_epoch = 100,
                         validation_data = test_set,
                         nb_val_samples = 1408)
"""

crtEpoch = 0
maxEpoch = 1500
while crtEpoch <= maxEpoch:
    classifier.fit_generator(training_set,
                             samples_per_epoch = 2880,
                             nb_epoch = 1,
                             validation_data = test_set,
                             nb_val_samples = 1408)

    classifier.save_weights(filepath = os.getcwd() + "/" + "savedModel-e" + str(crtEpoch) +".weights")

