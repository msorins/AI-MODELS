from io import BytesIO
from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha
import random
import os

num_train_data = 100000
num_test_data = int(num_train_data * 0.5)


def createFolders():
    try:
        os.mkdir(os.path.join(os.getcwd(), "dataset"))
        os.mkdir( os.path.join(os.getcwd(), "dataset", "training_set") )
        os.mkdir( os.path.join(os.getcwd(), "dataset", "test_set") )
    except:
        print("Some folders already exist")



def generateData(num, path):
    for crtImage in range(num):
        #nr_chars = random.randint(2, 6)
        nr_chars = 3

        chars = ''
        for _ in range(nr_chars):
            if _ == 0:
                digit = random.randint(1, 9)
            else:
                digit = random.randint(0, 9)

            chars += str(digit)

        image = ImageCaptcha(fonts=['fonts/captcha4.ttf'])\

        if( not os.path.isdir( os.path.join(path, chars)) ):
            os.mkdir( os.path.join( os.getcwd(), path, chars  ) )

        image.write(chars, os.path.join(path, chars, str(crtImage) + '.png'))

        if crtImage % 1000 == 0:
            print(crtImage, " images generated")


createFolders()
generateData(num_train_data, 'dataset/training_set')
generateData(num_test_data, 'dataset/test_set')
