import os
import sys
import zipfile

import boto3
import botocore.client
import botocore.exceptions
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
    # Establish connection with s3 via boto
    s3 = boto3.client('s3', config = botocore.client.Config(
        signature_version = botocore.UNSIGNED))
    s3_resource = boto3.resource('s3')

    # Setup progress bar
    try:
        ds_size = float(s3_resource.ObjectSummary(
            bucket_name = 'agdata-data',
            key = dataset_name + '.zip').size)
        pg = tqdm(
            total = ds_size, file = sys.stdout,
            desc = f"Downloading {dataset_name} "
                   f"(size = {round(ds_size / 1000000, 1)} MB)")
    except botocore.exceptions.ClientError as ce:
        if "Not Found" in str(ce):
            raise ValueError(
                f"The dataset '{dataset_name}' could not be found "
                f"in the bucket, perhaps it has not been uploaded "
                f"yet. Please report this issue.")
        raise ce

    # File path of zipped dataset
    if dataset_name in dest_dir:
        dest_dir = os.path.dirname(dest_dir)
    os.makedirs(dest_dir, exist_ok = True)
    dataset_download_path = os.path.join(
        dest_dir, dataset_name + '.zip')

    # Upload data to agdata-data bucket
    try:
        with open(dataset_download_path, 'wb') as data:
            s3.download_fileobj(Bucket = 'agdata-data',
                                Key = dataset_name + '.zip',
                                Fileobj = data,
                                Callback = lambda x: pg.update(x))
        pg.close()
    except BaseException as e:
        pg.close()
        if os.path.exists(dataset_download_path):
            os.remove(dataset_download_path)
        raise e

    # Unzip downloaded dataset
    with zipfile.ZipFile(dataset_download_path, 'r') as z:
        z.printdir()
        print('Extracting files...')
        z.extractall(path = dest_dir)
        print('Done!')

    # Delete zipped file
    if os.path.exists(dataset_download_path):
        os.remove(dataset_download_path)
