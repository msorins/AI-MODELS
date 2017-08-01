import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3" # Will use only 2nd, 3rd and 4th gpu

#Part 1 -> Importing
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Part 2 -> Initialise the CNN

classifier = Sequential()

# Step 1 -> Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 -> Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(rate = 0.2))

# Step 3 -> Convolution
classifier.add(Convolution2D(32, 3, 3,  activation = 'relu'))

# Step 4 -> Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(rate = 0.2))

classifier.add(Convolution2D(32, 3, 3,  activation = 'relu'))

# Step 4 -> Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 5 -> Flattening
classifier.add(Flatten())

# Step 6 -> Full connection
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Part 3 -> Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 4 -> Get the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#Part 5 -> fit the  NN
classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 20,
                         validation_data = test_set,
                         nb_val_samples = 2000)

classifier.save(filepath = os.getcwd() + "/" + "savedModel.data")