import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import imread


def file_as_matrix(path):
    as_arr = imread(path)
    im = Image.fromarray(as_arr)
    im.show()
    return 0


if __name__ == '__main__':
    x = file_as_matrix('liz.jpg')
    print(123123)
