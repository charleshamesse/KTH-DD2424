import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import imread
from inception_score import get_inception_score

DATA_DIR = '../dataset/photos_108/'

'''
Example output:
Inception score:	 1.5295098 +- 0.06244182
Excluded  16 / 30770 pictures
'''

if __name__ == '__main__':
    print("Loading dataset..")
    pictures = os.listdir(DATA_DIR)
    pictures_as_matrices = []
    excluded_count = 0
    for i, picture in enumerate(pictures):
        if i > 0 and i%2000 == 0:
            print("\tpicture", i)
        picture_path = os.path.join(DATA_DIR, picture)
        picture_mat = imread(picture_path)
        picture_final = np.array(picture_mat)

        # Some images don't have the right shape
        if picture_final.shape == (108, 108, 3):
            pictures_as_matrices.append(picture_final)
        else:
            excluded_count += 1
            print(picture, "is not correct:", picture_final.shape)

    print("Computing score..")
    with tf.Session() as sess:
        mu, sigma = get_inception_score(pictures_as_matrices, sess)    
    print("Inception score:\t", mu, "+-", sigma)

    print("Excluded ", excluded_count, "/", i+1, "pictures")