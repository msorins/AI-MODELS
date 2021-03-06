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
num_images = 20000
test_num_images = int(0.5 * num_images)

def addSquare(fig):
    ax = fig.add_subplot(111, aspect = 'equal')

    x = min(random.random(), 0.9)
    y = min(random.random(), 0.9)

    while True:
        width_v = random.random() / 5
        height_v = random.random() / 5
        if x + width_v  <= 1 and y + height_v <= 1:
            break

    ax.add_patch(patches.Rectangle(
        (x, y, 0.9),# (x, y)
         width = width_v,
         height = height_v,
         color= "green"
        )
    )



    # Return auxiliary info to folder
    return str(x) + "," + str(y + height_v) + "," + str(x + width_v) + "," + str(y)

def addCircle(fig):
    ax = fig.add_subplot(111, aspect='equal')

    x = random.random()
    y = random.random()
    while True:
        radius_v = max(0.1, random.random() / 8)
        if x + radius_v <= 1 and x - radius_v >= 0 and y + radius_v <= 1 and y - radius_v >= 0:
            break

    ax.add_patch( patches.Circle(
        (x, y),
        radius = radius_v
        )
    )

    #Returnauxiliary info
    return str(x-radius_v) + "," + str(y+radius_v) + "," + str(x+radius_v) + "," + str(y-radius_v)

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

def addTriangle(fig):
    ax = fig.add_subplot(111, aspect='equal')

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
    patchX.set_color("red")

    ax.add_patch(patchX)


    #Return auxiliary info
    return str(min(A_x, B_x, C_x)) + "," + str(max(A_y, B_y, C_y)) + "," + str(max(A_x, B_x, C_x)) + "," + str(min(A_y, B_y, C_y))


def createDirectories():
    try:
        os.mkdir(os.getcwd() + "/datasetv2")
        os.mkdir(os.getcwd() + "/datasetv2/training_set")
        os.mkdir(os.getcwd() + "/datasetv2/test_set")
    except:
        print("Folders already existing")

createDirectories()

def generateImages(num_images, path):
    for crt in range(num_images):

        # Create the image
        fig = plt.figure()
        fig.set_size_inches(10, 10)

        # Fill the image with shapes
        auxInfo = []
        for _ in range(random.randint(0, 4)):
            auxInfo += addTriangle(fig).split(",")
        for _ in range(random.randint(0, 4)):
            auxInfo += addCircle(fig).split(",")
        for _ in range(random.randint(0, 4)):
            auxInfo += addSquare(fig).split(",")

        if not len(auxInfo):
            continue

        # Save the image
        plt.tight_layout()
        plt.axis('off')
        fig.savefig(os.getcwd() + "/datasetv2/"+ path + "/" + str(crt) + '.png', bbox_inches="tight", dpi=100)


        #Save auxiliary info to folder
        f = open(os.getcwd() + "/datasetv2/"+ path +"/" + str(crt) + '.csv', 'w')
        f.write(",".join(auxInfo))
        f.close()


        #Save an auxiliary image with bounding boxes over the objects
        for i in range(len(auxInfo)):
            auxInfo[i] = float(auxInfo[i])

        i = 0
        while i < len(auxInfo):
            ax = fig.add_subplot(111, aspect='equal')
            fig.set_size_inches(10, 10)

            Path = patches.Path
            path_data = [
                (Path.MOVETO, [auxInfo[i], auxInfo[i+1]]),
                (Path.LINETO, [auxInfo[i+2], auxInfo[i+1]]),
                (Path.LINETO, [auxInfo[i+2], auxInfo[i+3]]),
                (Path.LINETO, [auxInfo[i], auxInfo[i+3]]),
                (Path.LINETO, [auxInfo[i], auxInfo[i+1]])
            ]

            codes, verts = zip(*path_data)
            pathX = patches.Path(verts, codes)
            patchX = patches.PathPatch(pathX)
            patchX.set_color("black")
            patchX.set_fill(None)
            patchX.set_linewidth(3)

            ax.add_patch(patchX)

            #Go to the next iteration
            i += 4

        # Save the image
        fig.savefig(os.getcwd() + "/datasetv2/"+ path +"/" + str(crt) + '_res.png', bbox_inches="tight", dpi=100)

        plt.close(fig)


generateImages(num_images, "training_set")
generateImages(test_num_images, "test_set")