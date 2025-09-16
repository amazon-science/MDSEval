import json
import requests
import os

def save_image_from_url(image_url, file_path):
    """
    Saves an image from a URL to a local file.

    Args:
        image_url (str): The URL of the image.
        file_path (str): The local path to save the image.
    """
    try:
        # check if the image already exists
        if os.path.exists(file_path):
            return True
        # Send a GET request to the URL
        response = requests.get(image_url)
        # Check if the request was successful
        response.raise_for_status()
        # Open the file in binary write mode and save the image content
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":

    data_path = 'MDSEval_data.json'

    # load json file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # extract all image urls and paths
    image_urls = []
    image_paths = []
    for dp in data:
        for img in dp['images']:
            image_urls.append(img['image_url'])
            image_paths.append(img['image_path'])


    # create image folders if not exist
    image_folders = ['images', 'images/PhotoChat', 'images/DialogCC']
    for path in image_folders:
        if not os.path.exists(path):
            os.mkdir(path)

    # download and save images
    for url, path in zip(image_urls, image_paths):
        save_image_from_url(url, path)      # Some images may fail to download, you can run this cell multiple times to download all images.