# Copyright 2021 UC Davis Plant AI and Biophysics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import zipfile

from agml.utils.data import (
    load_public_sources, maybe_you_meant, copyright_print
)
from agml.utils.logging import tqdm, log


def download_dataset(dataset_name, dest_dir, redownload = False):
    """
    Downloads a dataset from the agdata-data s3 file storage.

    Parameters
    ----------
    dataset_name : str
        name of dataset to download
    dest_dir : str
        path for saving downloaded dataset
    redownload : bool
        whether to re-download the dataset
    """
    import requests

    # Validate dataset name
    source_info = load_public_sources()
    if dataset_name not in source_info.keys():
        if dataset_name.replace('-', '_') not in source_info.keys():
            msg = f"Received invalid public source: '{dataset_name}'."
            msg = maybe_you_meant(dataset_name, msg)
            raise ValueError(msg)
        else:
            log(f"Interpreted dataset '{dataset_name}' as "
                f"'{dataset_name.replace('-', '_')}.'")
            dataset_name = dataset_name.replace('-', '_')

    # Connect to S3 and generate unsigned URL for bucket object
    url = f"https://agdata-data.s3.us-west-1.amazonaws.com/datasets/{dataset_name}.zip"

    # Check if dataset already exists
    if dataset_name in dest_dir:
        dest_dir = os.path.dirname(dest_dir)
    exist_dir = os.path.join(dest_dir, dataset_name)
    if not redownload:
        if os.path.exists(exist_dir):
            log(f"Dataset '{dataset_name}' already exists "
                f"in '{exist_dir}', skipping download.")
            return
    elif os.path.exists(exist_dir) and redownload:
        shutil.rmtree(exist_dir)

    # File path of zipped dataset
    os.makedirs(dest_dir, exist_ok = True)
    dataset_download_path = os.path.join(
        dest_dir, dataset_name + '.zip')

    # Download object from bucket
    try:
        with requests.Session() as sess:
            r = sess.get(url, stream = True)
            r.raise_for_status()
            content_size = int(r.headers['Content-Length'])
            sz = round(content_size / 1000000, 1)
            size_print = f"{sz} MB"
            if sz > 1000:
                size_print = f"{round(sz / 1000, 2)} GB"
            pg = tqdm(total = content_size,
                      desc = f"Downloading {dataset_name} "
                             f"(size = {size_print})")
            with open(dataset_download_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size = 8192):
                    f.write(chunk)
                    pg.update(8192)
            pg.close()
    except BaseException as e:
        try:
            pg.close()
        except (NameError, UnboundLocalError):
            pass
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

    # Print dataset copyright info
    copyright_print(dataset_name, os.path.splitext(dataset_download_path)[0])


def download_model(dataset_name, dest_dir, redownload = False):
    """
    Downloads a model from the agdata-data s3 file storage.

    Parameters
    ----------
    dataset_name : str
        name of the dataset for the model to download
    dest_dir : str
        path for saving downloaded model
    redownload : bool
        whether to re-download the model
    """
    import requests

    # Validate dataset name
    source_info = load_public_sources()
    if dataset_name not in source_info.keys():
        if dataset_name.replace('-', '_') not in source_info.keys():
            msg = f"Received invalid public source: '{dataset_name}'."
            msg = maybe_you_meant(dataset_name, msg)
            raise ValueError(msg)
        else:
            log(f"Interpreted dataset '{dataset_name}' as "
                f"'{dataset_name.replace('-', '_')}.'")
            dataset_name = dataset_name.replace('-', '_')

    # Connect to S3 and generate unsigned URL for bucket object
    url = f"https://agdata-data.s3.us-west-1.amazonaws.com/models/{dataset_name}.pth"

    # Check if model already exists
    if dataset_name in dest_dir:
        dest_dir = os.path.dirname(dest_dir)
    exist_dir = os.path.join(dest_dir, dataset_name)
    if not redownload:
        if os.path.exists(exist_dir):
            log(f"Model '{dataset_name}' already exists "
                f"in '{exist_dir}', skipping download.")
            return
    elif os.path.exists(exist_dir) and redownload:
        shutil.rmtree(exist_dir)

    # File path of model
    os.makedirs(dest_dir, exist_ok = True)
    model_download_path = os.path.join(dest_dir, dataset_name + '.pth')

    # Download object from bucket
    try:
        with requests.Session() as sess:
            r = sess.get(url, stream = True)
            r.raise_for_status()
            content_size = int(r.headers['Content-Length'])
            sz = round(content_size / 1000000, 1)
            size_print = f"{sz} MB"
            if sz > 1000:
                size_print = f"{round(sz / 1000, 2)} GB"
            pg = tqdm(total = content_size,
                      desc = f"Downloading {dataset_name} "
                             f"(size = {size_print})")
            with open(model_download_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size = 8192):
                    f.write(chunk)
                    pg.update(8192)
            pg.close()
    except BaseException as e:
        try:
            pg.close()
        except (NameError, UnboundLocalError):
            pass
        if os.path.exists(model_download_path):
            os.remove(model_download_path)
        raise e

