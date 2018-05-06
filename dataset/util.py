import json
import requests
import urllib
import os
import numpy as np
from datetime import datetime

# Might need to create this folder
DATA_FOLDER = './photos/'

def remove_duplicates():
    downloaded_photos = os.listdir(DATA_FOLDER)
    k = 0
    photo_ids = []
    for photo in downloaded_photos:
        if len(photo.split('-')) < 2:
            photo_ids.append(photo.split('_')[1][:-4])
            k = k+1
    
    
    j = 0
    for photo in downloaded_photos:
        if len(photo.split('-')) >= 2:
            photo_id = photo.split('_')[-1][:-4]
            if photo_id in photo_ids:
                j += 1
    
    print(j)
# Entry point
if __name__ == '__main__':
    remove_duplicates()