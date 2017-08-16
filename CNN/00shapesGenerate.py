import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import math
# Create images with random rectangles and bounding boxes.
#
num_circles = 600
num_squares = 600
num_triangles = 600
num_crt = 0
test_num_circles = int(0.2 * num_circles)
test_num_squares = int(0.2 * num_squares)
test_num_triangles = int(0.2 * num_triangles)

def createSquare(name, path):
    fig1 = plt.figure()
    fig1.set_size_inches(4, 4)
    ax1 = fig1.add_subplot(111, aspect = 'equal')

    x = min(random.random(), 0.9)
    y = min(random.random(), 0.9)

    while True:
        width_v = random.random() / 5
        height_v = random.random() / 5
        if x + width_v  <= 1 and y + height_v <= 1:
            break

    ax1.add_patch(patches.Rectangle(
        (x, y, 0.9),# (x, y)
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
    f = open(os.getcwd() + "/dataset/" +path + "/" + name + '.csv', 'w')
    f.write(str(x) + "," + str(y + height_v) + "," + str(x + width_v) + "," + str(y))
    f.close()

def createCircle(name, path):
    fig1 = plt.figure()
    fig1.set_size_inches(4, 4)
    ax1 = fig1.add_subplot(111, aspect='equal')

    x = random.random()
    y = random.random()
    while True:
        radius_v = max(0.1, random.random() / 8)
        if x + radius_v <= 1 and x - radius_v >= 0 and y + radius_v <= 1 and y - radius_v >= 0:
            break

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
    f = open(os.getcwd() + "/dataset/" +path + "/" +name +'.csv', 'w')
    f.write(str(x-radius_v) + "," + str(y+radius_v) + "," + str(x+radius_v) + "," + str(y-radius_v))
    f.close()

def area(a, b, c):
    def distance(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    side_a = distance(a, b)
    side_b = distance(b, c)
    side_c = distance(c, a)
    s = 0.5 * ( side_a + side_b + side_c)
    return math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))

def dist(a, b):
    return math.sqrt( ( a[0] - b[0] ) * ( a[0] - b[0] ) + ( a[1] - b[1] ) * ( a[1] - b[1] ) )

def createTriangle(name, path):
    fig1 = plt.figure()
    fig1.set_size_inches(4, 4)
    ax1 = fig1.add_subplot(111, aspect='equal')

    while True:
        A_x = random.random()
        A_y = random.random()
        B_x = random.random()
        B_y = random.random()
        C_x = random.random()
        C_y = random.random()

        if dist([A_x, A_y], [B_x, B_y]) <= 0.3 and dist([A_x, A_y], [C_x, C_y]) <= 0.3 and dist([B_x, B_y], [C_x, C_y]) <= 0.3 and area([A_x, A_y], [B_x, B_y], [C_x, C_y]) >= 0.01:
            break

    Path = patches.Path
    path_data = [
        (Path.MOVETO, [A_x, A_y]),
        (Path.LINETO, [B_x, B_y]),
        (Path.LINETO, [C_x, C_y]),
        (Path.CLOSEPOLY, [C_x, C_y])
    ]
    codes, verts = zip(*path_data)
    pathX = patches.Path(verts, codes)
    patchX = patches.PathPatch(pathX)

    ax1.add_patch(patchX)

    #Save image to folder
    plt.tight_layout()
    plt.axis('off')
    fig1.savefig(os.getcwd() + "/dataset/" + path + "/" + name + '.png', bbox_inches="tight", dpi=100)
    plt.close(fig1)

    #Save auxiliary info to folder
    f = open(os.getcwd() + "/dataset/" +path + "/" +name +'.csv', 'w')
    f.write( str(min(A_x, B_x, C_x)) + "," + str(max(A_y, B_y, C_y)) + "," + str(max(A_x, B_x, C_x)) + "," + str(min(A_y, B_y, C_y)) )
    f.close()



def createDirectories():
    try:
        os.mkdir(os.getcwd() + "/dataset")
        os.mkdir(os.getcwd() + "/dataset/training_set")
        os.mkdir(os.getcwd() + "/dataset/test_set")
        os.mkdir(os.getcwd() + "/dataset/test_set/circle")
        os.mkdir(os.getcwd() + "/dataset/test_set/square")
        os.mkdir(os.getcwd() + "/dataset/test_set/triangle")
        os.mkdir(os.getcwd() + "/dataset/training_set/square")
        os.mkdir(os.getcwd() + "/dataset/training_set/circle")
        os.mkdir(os.getcwd() + "/dataset/training_set/triangle")
    except:
        print("Folders already existing")

createDirectories()

for _ in range(num_circles):
    createCircle(str(num_crt), "training_set/circle")
    num_crt += 1

for _ in range(num_squares):
    createSquare(str(num_crt), "training_set/square")
    num_crt += 1

for _ in range(num_triangles):
    createTriangle(str(num_crt), "training_set/triangle")
    num_crt += 1

for _ in range(test_num_circles):
    createCircle(str(num_crt), "test_set/circle")
    num_crt += 1

for _ in range(test_num_squares):
    createSquare(str(num_crt), "test_set/square")
    num_crt += 1

for _ in range(test_num_triangles):
    createTriangle(str(num_crt), "test_set/triangle")
    num_crt += 1



