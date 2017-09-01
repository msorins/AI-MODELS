#98% percentage
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" # Will use only 2nd, 3rd and 4th gpu

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

# Step 1 -> Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (256, 256, 3), activation = 'relu'))

# Step 2 -> Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 -> Convolution
classifier.add(Convolution2D(32, 3, 3,  activation = 'relu'))

# Step 4 -> Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 -> Convolution
classifier.add(Convolution2D(32, 3, 3,  activation = 'relu'))

# Step 4 -> Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 5 -> Flattening
classifier.add(Flatten())

# Step 6 -> Full connection
classifier.add(Dense(output_dim = 256, activation = 'relu'))
#classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

#Part 3 -> Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Part 4 -> Get the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')
#Part 5 -> fit the  NN
classifier.fit_generator(training_set,
                         samples_per_epoch = 1800, #equal to the number of training data
                         nb_epoch = 28,
                         validation_data = test_set,
                         nb_val_samples = 720 #equal to the number of test data
                         )

#Part 6 -> save
pth = str( os.getcwd() + "/" + sys.argv[0] + ".model")
print("Path is: ", pth)
classifier.save(filepath = pth)
