import requests

def download_data(url, file_name):
    """
    Downloads data from a specified URL and saves it to a file.

    Args:
        url (str): The URL to download the data from.
        file_name (str): The name of the file to save the data.

    Returns:
        None
    """
    response = requests.get(url)
    with open(file_name, 'w') as f:
        f.write(response.text)
