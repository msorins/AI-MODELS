import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import math

#Colors for the shaoe
cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'
}

# Create images with random rectangles and bounding boxes.
#

num_images = 2000
test_num_images = int(0.5 * num_images)

def addSquare(fig):
    ax = fig.add_subplot(111, aspect = 'equal')

    x = max(0.1, min(random.random(), 0.9))
    y = max(0.1, min(random.random(), 0.9))

    while True:
        width_v = random.random() / 5
        height_v = random.random() / 5
        if x + width_v  <= 1 and y + height_v <= 1:
            break

    #Choose a color
    colorname, colorcode = random.choice(list(cnames.items()))

    ax.add_patch(patches.Rectangle(
        (x, y, 0.9),# (x, y)
         width = width_v,
         height = height_v,
         color= colorcode
        )
    )



    # Return auxiliary info to folder
    return str(x) + "," + str(y + height_v) + "," + str(x + width_v) + "," + str(y)

def addCircle(fig):
    ax = fig.add_subplot(111, aspect='equal')


    while True:
        x = random.random()
        y = random.random()
        radius_v = max(0.1, random.random() / 2)

        if x + radius_v <= 1 and x - radius_v >= 0 and y + radius_v <= 1 and y - radius_v >= 0:
            break

    # Choose a color
    colorname, colorcode = random.choice(list(cnames.items()))

    ax.add_patch( patches.Circle(
        (x, y),
        radius = radius_v,
        color = colorcode
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

    # Choose a color
    colorname, colorcode = random.choice(list(cnames.items()))
    patchX.set_color(colorcode)

    ax.add_patch(patchX)


    #Return auxiliary info
    return str(min(A_x, B_x, C_x)) + "," + str(max(A_y, B_y, C_y)) + "," + str(max(A_x, B_x, C_x)) + "," + str(min(A_y, B_y, C_y))


def createDirectories():
    try:
        os.mkdir(os.getcwd() + "/datasetv4")
        os.mkdir(os.getcwd() + "/datasetv4/training_set")
        os.mkdir(os.getcwd() + "/datasetv4/test_set")
        for i in range(1, 33):
            os.mkdir(os.getcwd() + "/datasetv4/training_set/" + str(i) + "/")
            os.mkdir(os.getcwd() + "/datasetv4/test_set/" + str(i) + "/")
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
        totalImg = 0
        for _ in range(random.randint(0, 2)):
            auxInfo += addTriangle(fig).split(",")
            totalImg += 1
        for _ in range(random.randint(0, 2)):
            auxInfo += addCircle(fig).split(",")
            totalImg += 1
        for _ in range(random.randint(0, 2)):
            auxInfo += addSquare(fig).split(",")
            totalImg += 1

        if not len(auxInfo):
            continue

        # Save the image
        plt.tight_layout()
        plt.axis('off')
        fig.savefig(os.getcwd() + "/datasetv4/"+ path + "/" + str(totalImg) + "/" + str(crt) + '.png', bbox_inches="tight", dpi=100)


        #Save auxiliary info to folder
        """
        f = open(os.getcwd() + "/datasetv4/"+ path +"/" + str(totalImg) + "/" + str(crt) + '.csv', 'w')
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
        fig.savefig(os.getcwd() + "/datasetv4/"+ path +"/" + str(totalImg) + "/" + str(crt) + '_res.png', bbox_inches="tight", dpi=100)
        plt.close(fig)
        print("Generated to: ", os.getcwd() + "/datasetv4/"+ path +"/" + str(totalImg) + "/" + str(crt) + '_res.png')
        """


generateImages(num_images, "training_set")
generateImages(test_num_images, "test_set")