#!/usr/bin/python
import PIL
from PIL import Image
import os, sys

path = "/home/saurabh/Test1/"
dirs = os.listdir( path )
#print path
def resize():
    for item in dirs:
    	#print os.path.isfile(path+item)
    	#print path+item
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            #print f
            imResize = im.resize((244,244), PIL.Image.ANTIALIAS)
            imResize = imResize.convert('L')
            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

resize()