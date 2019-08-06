import os
import urllib.request
from zipfile import ZipFile

def download_zip(root_dir, url):
    dataset_name = url.split("/")[-1]
    assert ".zip" == dataset_name[-4:]
    dataset_name = dataset_name[:-4]
    if not os.path.exists(os.path.join(root_dir, dataset_name)):
        urllib.request.urlretrieve(url, "{}.zip".format(dataset_name))
        # Create a ZipFile Object and load sample.zip in it
        with ZipFile('{}.zip'.format(dataset_name), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(root_dir)
        os.remove("{}.zip".format(dataset_name))
    else:
        print("Using buffered data.")
