import json
import requests
import urllib
import os
import sys
import scipy
import numpy as np
from scipy.ndimage import imread
from PIL import Image

# Might need to create this folder
DATA_FOLDER = './photos/'

DATA_FOLDER_OUT = './photos_108/'

size = 108, 108

def resize_photo(infile):
    outfile = infile.split('.jpg')[0] + '_108x108.jpg'
    try:
        im = Image.open(os.path.join(DATA_FOLDER, infile))
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(os.path.join(DATA_FOLDER_OUT, outfile), "JPEG")
    except IOError as e:
        print("Cannot create thumbnail for '%s'" % infile, e)

def resize_photos():
    photos = os.listdir(DATA_FOLDER)
    k = 0
    for photo in photos:
        if k % 100 == 0:
            print(k)
        resize_photo(photo)
        k += 1

def pack_photos():
    photos = os.listdir(DATA_FOLDER_OUT)
    X = []
    k = 0
    for photo in photos:
        if photo.split('_')[-1] == '108x108.jpg':
            if k % 100 == 0:
                print(k)
            x = imread(os.path.join(DATA_FOLDER, photo))
            X.append(x)
            k += 1

    X = np.array(X)
    print(X.shape)
    np.save('animals.npy', X)

# Entry point
if __name__ == '__main__':
    resize_photos()
    #pack_photos()