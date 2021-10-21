import os
import sys
import zipfile

import requests
from tqdm import tqdm

def download_dataset(dataset_name, dest_dir):
    """
    Downloads dataset from agdata-data s3 file storage.

    Parameters
    ----------
    dataset_name : str
        name of dataset to download
    dest_dir : str
        path for saving downloaded dataset
    """
    # Connect to S3 and generate unsigned URL for bucket object
    url = f"https://agdata-data.s3.us-west-1.amazonaws.com/{dataset_name}.zip"

    # File path of zipped dataset
    if dataset_name in dest_dir:
        dest_dir = os.path.dirname(dest_dir)
    os.makedirs(dest_dir, exist_ok = True)
    dataset_download_path = os.path.join(
        dest_dir, dataset_name + '.zip')

    # Download object from bucket
    try:
        with requests.Session() as sess:
            r = sess.get(url, stream = True)
            r.raise_for_status()
            content_size = int(r.headers['Content-Length'])
            pg = tqdm(total = content_size, file = sys.stdout,
                      desc = f"Downloading {dataset_name} "
                             f"(size = {round(content_size/ 1000000, 1)} MB)")
            with open(dataset_download_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size = 8192):
                    f.write(chunk)
                    pg.update(8192)
            pg.close()
    except BaseException as e:
        pg.close()
        if os.path.exists(dataset_download_path):
            os.remove(dataset_download_path)
        raise e

    # Unzip downloaded dataset
    with zipfile.ZipFile(dataset_download_path, 'r') as z:
        print(f'[AgML Download]: Extracting files for {dataset_name}... ', end = '')
        z.extractall(path = dest_dir)
        print('Done!')

    # Delete zipped file
    if os.path.exists(dataset_download_path):
        os.remove(dataset_download_path)
