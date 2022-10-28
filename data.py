import os
import zipfile
import requests


def dl_data(dir='data'):
    url = 'https://zenodo.org/record/7070963/files/Modescaling_raw_data.zip?download=1'

    # Download the data
    r = requests.get(url, allow_redirects=True)
    open('data.zip', 'wb').write(r.content)

    # Unzip the data
    with zipfile.ZipFile('data.zip', 'r') as zip_ref:
        zip_ref.extractall(dir)

    # Remove the zip file
    os.remove('data.zip')


if __name__ == "__main__":
    if not os.path.exists('data'):
        os.mkdir('data')
        dl_data()
    else:
        print('Data already downloaded')