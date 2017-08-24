import os

import keras
from keras import callbacks
from keras.callbacks import ModelCheckpoint, Callback

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3" # Will use only 2nd, 3rd and 4th gpu

#Part 1 -> Importing
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import sys

#Part 2 -> Initialise the CNN

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Convolution2D(64, 3, 3,  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(64, 3, 3,  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(64, 3, 3,  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

# Step 6 -> Full connection
classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dense(output_dim = 1024, activation = 'relu'))
classifier.add(Dense(output_dim = 1024, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'linear'))

#Part 3 -> Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

#Part 4 -> Get the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory('datasetv6/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory('datasetv6/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'sparse')

class Save(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        filepath = "model-improvement-" + sys.argv[0] + ".hdf5"
        classifier.save(filepath)


# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-"+ sys.argv[0] +".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
save = Save()
callbacks_list = [checkpoint, save]



#Part 5 -> fit the  NN
classifier.fit_generator(training_set,
                         samples_per_epoch = 992,
                         nb_epoch = 40,
                         validation_data = test_set,
                         nb_val_samples = 480,
                         callbacks = callbacks_list)

classifier.save(filepath = os.getcwd() + "/" + "savedModel.data")