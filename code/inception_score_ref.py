import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import imread
from inception_score import get_inception_score
from keras.datasets import cifar10

DATA_DIR = '../dataset/photos_108/'
DATASET_NAME = 'cifar10'

'''
Example output using 10 splits 30K:
Inception score:	 1.5295098 +- 0.06244182
Excluded  16 / 30770 pictures

Example output using 10 splits 15K:
Inception score:	 1.483624 +- 0.10410255
Excluded  10 / 15000 pictures

Example output using 10 splits 3K:
Inception score:	 1.2856357 +- 0.11991055
Excluded  2 / 3000 pictures
'''

if __name__ == '__main__':
    print("Loading dataset..")
    excluded_count = 0

    if DATASET_NAME is 'cifar10':
        image_shape = (32,32,3)
        (x_train, _), (_, _) = cifar10.load_data()
        limit = 1000# None # 50K images
        pictures_as_matrices = [x for x in x_train[0:limit]]
        i = len(pictures_as_matrices)
    else:
        image_shape = (108,108,3)
        pictures = os.listdir(DATA_DIR)[0:15000]
        pictures_as_matrices = []
        for i, picture in enumerate(pictures):
            if i > 0 and i%2000 == 0:
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
        mu, sigma = get_inception_score(pictures_as_matrices, sess, splits=10)    
    print("Inception score:\t", mu, "+-", sigma)

    print("Excluded ", excluded_count, "/", i+1, "pictures")