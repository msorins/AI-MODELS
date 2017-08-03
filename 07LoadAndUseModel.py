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
img_cols, img_rows = 64, 64

def get_im(path):
    # Load as grayscale
    print("Path: ", path)
    img = cv2.imread(path)
    # Reduce size
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
        img = numpy.array(img)
        img = numpy.reshape(img, (1, img_cols, img_rows, 3))
        prediction = model.predict(x = img)
        prediction = prediction[0][0]

        #Scatter the result
        plt.scatter(x=prediction * 1000000000, y = 0)

        #Save the result
        res += str(fl) + " => prediction: " + str( prediction ) + "\n"

        crt += 1
        if crt >= 350:
            break


    plt.show()

    return res

def save_string_to_file(string, path):
    f = open(path, 'w')
    f.write(string)
    f.close()

def main():
    # Load the model
    model = load_model('07isMushroom.h5')

    # Load each image folder and predict
    predictionsGoodMushrooms = load_images_predict(model, os.path.join('mushrooms', 'GoodMushrooms', '*.jpg'))
    save_string_to_file(predictionsGoodMushrooms, './GoodMushroomsPredictions.txt')

    # Load each image folder and predict
    predictionsBadMushrooms = load_images_predict(model, os.path.join('mushrooms', 'BadMushrooms', '*.jpg'))
    save_string_to_file(predictionsBadMushrooms, './BadMushroomsPredictions.txt')
    
#Execute the main function    
main()