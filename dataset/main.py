import json
import requests
import urllib
import os

# API info
API_KEY = '78d7c05a38eb68608887093fd60516b6'
API_URL = 'https://api.flickr.com/services/rest/'

# Might need to create this folder
DATA_FOLDER = './photos/'


# Reconstruct photo url
def get_photo_url(farm_id, server_id, photo_id, secret):
    return 'https://farm' + str(farm_id) + '.staticflickr.com/' + str(server_id) + '/' + str(photo_id) + '_' + str(secret) + '_q.jpg'

# Save image from url to disk
def save_image(url, fname):
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
        return True
    return False

#  Download all photos given a query text 
def download_photos(query):
    parameters = {
        "api_key": API_KEY,
        "format": "json",
        "sort": "relevance", 
        "method": "flickr.photos.search",
        "text": query
    }
    response = requests.get(API_URL, parameters)

    # Print the status code of the response.
    if response.status_code == 200:
        # Response is wrapped in "jsonFlickrApi()" so we remove it before parsing
        json_data = json.loads(response.text[14:-1])

        # For all photos
        for photo in json_data['photos']['photo']:
            # Build url and filename
            url = get_photo_url(photo["farm"], photo["server"], photo["id"], photo["secret"])
            filename = query + "_" + str(photo["id"]) + '.jpg'
            # Try to save it
            try:
                save_image(url, os.path.join(DATA_FOLDER, filename))
            except Exception as e:
                print("Error: could not download " + filename, e)

    else:
        print("API call error")

# Entry point
if __name__ == '__main__':
    download_photos("snake")