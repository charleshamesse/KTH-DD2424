import requests


'''
Example request
https://api.flickr.com/services/rest/?api_key=78d7c05a38eb68608887093fd60516b6&method=flickr.photos.search&sort=relevance&text=snake&per_page=500
'''
API_KEY = '78d7c05a38eb68608887093fd60516b6'
BASE_URL = 'https://api.flickr.com/services/rest/'


# Make a get request to get the latest position of the international space station from the opennotify api.
response = requests.get(
    BASE_URL + "?api_key=" + API_KEY \
    + "&method=flickr.photos.search" \
    + "&sort=relevance" \
    + "&text=snake"
)

# Print the status code of the response.
print(response.status_code)