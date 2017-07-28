import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

# Create images with random rectangles and bounding boxes.
#
num_circles = 500
num_squares = 500
num_crt = 0
test_num_circles = int(0.2 * num_circles)
test_num_squares = int(0.2 * num_squares)

def createSquare(name, path):
    fig1 = plt.figure()
    fig1.set_size_inches(4, 4)
    ax1 = fig1.add_subplot(111, aspect = 'equal')

    x = min(random.random(), 0.9)
    y = min(random.random(), 0.9)
    width_v = width= random.random() / 5
    height_v=  random.random() / 5

    ax1.add_patch(patches.Rectangle(
        (x, min(random.random(), 0.9)),# (x, y)
         width = width_v,
         height = height_v
         )
    )

    # Save image to folder
    plt.tight_layout()
    plt.axis('off')
    fig1.savefig(os.getcwd() + "/dataset/" +path + "/" + name + '.png', bbox_inches="tight", dpi=100)
    plt.close(fig1)

    # Save auxiliary info to folder
    f = open(os.getcwd() + "/datainfo/" +path + "/" + name + '.csv', 'w')
    f.write(str(x) + "," + str(y + height_v) + "," + str(x + width_v) + "," + str(y))
    f.close()

def createCircle(name, path):
    fig1 = plt.figure()
    fig1.set_size_inches(4, 4)
    ax1 = fig1.add_subplot(111, aspect='equal')

    x = random.random()
    y = random.random()
    radius_v = random.random() / 8

    ax1.add_patch( patches.Circle(
        (x, y),
        radius = radius_v
        )
    )

    #Save image to folder
    plt.tight_layout()
    plt.axis('off')
    fig1.savefig(os.getcwd() + "/dataset/" + path + "/" + name + '.png', bbox_inches="tight", dpi=100)
    plt.close(fig1)

    #Save auxiliary info to folder
    f = open(os.getcwd() + "/datainfo/" +path + "/" +name +'.csv', 'w')
    f.write(str(x-radius_v) + "," + str(y+radius_v) + "," + str(x+radius_v) + "," + str(y-radius_v))
    f.close()


def createDirectories():
    try:
        os.mkdir(os.getcwd() + "/dataset")
        os.mkdir(os.getcwd() + "/dataset/training_set")
        os.mkdir(os.getcwd() + "/dataset/test_set")
        os.mkdir(os.getcwd() + "/dataset/test_set/circle")
        os.mkdir(os.getcwd() + "/dataset/test_set/square")
        os.mkdir(os.getcwd() + "/dataset/training_set/square")
        os.mkdir(os.getcwd() + "/dataset/training_set/circle")

        os.mkdir(os.getcwd() + "/datainfo")
        os.mkdir(os.getcwd() + "/datainfo/training_set")
        os.mkdir(os.getcwd() + "/datainfo/test_set")
        os.mkdir(os.getcwd() + "/datainfo/test_set/circle")
        os.mkdir(os.getcwd() + "/datainfo/test_set/square")
        os.mkdir(os.getcwd() + "/datainfo/training_set/square")
        os.mkdir(os.getcwd() + "/datainfo/training_set/circle")
    except:
        print("Folders already existing")

createDirectories()

for _ in range(num_circles):
    createCircle(str(num_crt), "training_set/circle")
    num_crt += 1

for _ in range(num_squares):
    createSquare(str(num_crt), "training_set/square")
    num_crt += 1

for _ in range(test_num_circles):
    createCircle(str(num_crt), "test_set/circle")
    num_crt += 1

for _ in range(test_num_squares):
    createSquare(str(num_crt), "test_set/square")
    num_crt += 1





