import os
from shutil import copyfile
from random import randint
import cv2

dim = 58, 62

for dirname, dirnames, filenames in os.walk('.'):
    if dirname == ".":
        continue
    for i in range(1, 6):
        src = dirname+"/image_000"+str(i)+".jpg"
        dest = "./mixed/"+str(randint(1, 1000))+".jpg"
        copyfile(src, dest)
        image = cv2.imread(dest)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale, dim, interpolation = cv2.INTER_AREA)
        low_res = "./lowres/"+str(randint(1, 1000))+".jpg"
        cv2.imwrite(low_res, resized)
