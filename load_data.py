import requests

# URL of the text file
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract the text data from the response
    text_data = response.text
    save_path = 'data.txt'
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print('File downloaded successfully:', save_path)
else:
    # Print an error message if the request was not successful
    print('Failed to retrieve data from the URL:', url)
