import glob
from keras.models import load_model
from numpy import genfromtxt
import cv2
import os
import h5py

#Global variables
cwd = os.getcwd()
img_cols, img_rows = 256, 256


def get_im(path):
    # Load as grayscale
    print("Path: ", path)
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def load_images(path):
    images = []
    files = glob.glob(path)

    for fl in files:
        if "res" in fl:
            continue

        fl = cwd + "/" + fl
        img = get_im(fl)
        images.append(img)

    return images

def evaluate_prediction(model, images):
    for crtImg in images:
        print( model.evaluate(crtImg) )

def main():
    # Load the model
    model = load_model('07isMushroom.h5')

    # Load each image folder and evaluate
    imgs = load_images(os.path.join('mushrooms', 'goodmushrooms', '*.png'))
    evaluate_prediction(model, imgs)

    
#Execute the main function    
main()