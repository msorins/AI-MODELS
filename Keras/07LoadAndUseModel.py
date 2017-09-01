import glob
from keras.models import load_model
from numpy import genfromtxt
import numpy
import cv2
import os
import h5py
from decimal import Decimal
import matplotlib.pyplot as plt

#Global variables
cwd = os.getcwd()
img_cols, img_rows = 128, 128

def get_im(path):
    #print("Path: ", path)
    img = cv2.imread(path)

    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_images_predict(model, path):
    res = ''
    files = glob.glob(path)

    crt = 0
    for fl in files:
        if "res" in fl:
            continue

        #Load the image
        fl = cwd + "/" + fl
        img = get_im(fl)

        #Predict a result
        img = numpy.array(img).astype('float32')
        img /= 255
        img = numpy.reshape(img, (1, img_cols, img_rows, 3))

        prediction = model.predict(x = img)
        prediction = prediction[0][0]

        #print(path + ": " + str(prediction) )
        #image = Image.open(path)
        #image.show()

        #Scatter the result
        #plt.scatter(x=prediction, y = 0)

        #Save the result
        res += str(fl) + " => prediction: " + str( prediction ) + "\n"

        #crt += 1
        #if crt >= 550:
        #    break

    plt.show()

    return res

def save_string_to_file(string, path):
    f = open(path, 'w')
    f.write(string)
    f.close()

def main():
    # Load the model
    model = load_model('12Captcha/12v4CNNLSTMModel-TrainCNNWeights-14-143922.4646.hdf5')

    # Load each image folder and predict
    predictionsGoodMushrooms = load_images_predict(model, os.path.join('mushrooms', 'GoodMushrooms', '*.jpg'))
    save_string_to_file(predictionsGoodMushrooms, './GoodMushroomsPredictions.txt')

    # Load each image folder and predict
    predictionsBadMushrooms = load_images_predict(model, os.path.join('mushrooms', 'BadMushrooms', '*.jpg'))
    save_string_to_file(predictionsBadMushrooms, './BadMushroomsPredictions.txt')
    
#Execute the main function    
main()