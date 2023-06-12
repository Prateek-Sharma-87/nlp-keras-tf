import urllib.request

url = 'https://storage.googleapis.com/learning-datasets/irish-lyrics-eof.txt'

# Method 1

urllib.request.urlretrieve(url , filename="irish_lyrics_eof.txt") # To download the file from `url` and save it locally under `file_name`


# Method 2 (Alternate method)

# local_filename, headers = urllib.request.urlretrieve(url)
# text = open(local_filename).read()

# with open('irish_lyrics_eof_check.txt', 'w') as f:
#     f.write(text)


# Method 3 (Alternate method)

# response = urllib.request.urlopen(url)
# data = response.read()          # a `bytes` object
# text = data.decode('utf-8')     # a `str`; this step can't be used if the data is binary

# with open('irish_lyrics_eof_check.txt', 'w') as f:
#     f.write(text)
