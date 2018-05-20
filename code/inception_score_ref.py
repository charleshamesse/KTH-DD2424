import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import imread
from inception_score import get_inception_score
from keras.datasets import cifar10

DATA_DIR = '../dataset/photos_108/'
DATASET_NAME = 'reptiles'#'cifar10'

'''
Reptiles:
5 splits, 1024 image
Inception score:	 10.2415905 +- 0.92625165

Cifar10:
5 splits, 1024 images
Inception score:	 9.792953 +- 0.36040735

5 splits, 512 images
Inception score:	 8.938408 +- 0.54188323
Excluded  0 / 513 pictures

5 splits, 256 images
Inception score:	 8.130996 +- 0.4138345

10 splits, 256 images
Inception score:	 6.717534 +- 0.5547067

10 splits, 512 images
Inception score:	 7.92 +- 0.??

10 splits, 1024 images
Inception score:	 8.947193 +- 0.4389134
'''

if __name__ == '__main__':
    print("Loading dataset..")
    excluded_count = 0

    if DATASET_NAME is 'cifar10':
        image_shape = (32,32,3)
        (x_train, _), (_, _) = cifar10.load_data()
        limit = 1024# None # 50K images
        pictures_as_matrices = [x for x in x_train[0:limit]]
        i = len(pictures_as_matrices)
    else:
        image_shape = (108,108,3)
        pictures = os.listdir(DATA_DIR)[0:1024]
        pictures_as_matrices = []
        for i, picture in enumerate(pictures):
            if i > 0 and i%200 == 0:
                print("\tpicture", i)
            picture_path = os.path.join(DATA_DIR, picture)
            picture_mat = imread(picture_path)
            picture_final = np.array(picture_mat)

            # Some images don't have the right shape
            if picture_final.shape == image_shape:
                pictures_as_matrices.append(picture_final)
            else:
                excluded_count += 1
                print(picture, "is not correct:", picture_final.shape)

    print("Computing score..")
    with tf.Session() as sess:
        mu, sigma = get_inception_score(pictures_as_matrices, sess, splits=5)    
    print("Inception score:\t", mu, "+-", sigma)

    print("Excluded ", excluded_count, "/", i+1, "pictures")