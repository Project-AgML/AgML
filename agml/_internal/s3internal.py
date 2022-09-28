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

"""
Internal class for interacting with the AgML public dataset S3 bucket.
"""

import os
import sys
import json
import zipfile
import warnings

import boto3
import botocore.exceptions

from agml.utils.logging import tqdm


class InternalAgMLS3API(object):
    """
    Internal API for interaction with the dataset S3 bucket.

    *Note*: This class should only be used by the AgML developers.
    """
    def __init__(self):
        # Read in metadata for data sources file
        self.data_srcs_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '_assets', 'public_datasources.json')
        with open(self.data_srcs_path) as f:
            self.data_srcs = json.load(f)

        # Define s3 bucket URI
        self.agdata_s3_uri = 'https://s3.us-west-1.amazonaws.com/agdata-data/'

        # Initialize attribute for storing dataset download path
        self.dataset_download_path = None

    @property
    def data_sources(self):
        """Returns a list of AgML public data sources."""
        return list(self.data_srcs.keys())

    def upload_dataset(self, dataset_name, dataset_dir):
        """
        Uploads dataset to agdata-data s3 file storage.
        
        Parameters
        ----------
        dataset_name : str
            name of dataset (without '.zip') -- for list of datasets run self.data_srcs.keys()
        dataset_dir : str
            path to directory where dataset is stored
        """
        # Establish connection with s3 via boto
        self.s3 = boto3.client('s3')

        # Setup progress bar
        self.pg = tqdm(
            total=os.stat(os.path.abspath(os.path.join(dataset_dir, dataset_name + '.zip'))).st_size,
            file=sys.stdout, desc=f"Uploading {dataset_name}")

        # Upload data to agdata-data bucket
        try:
            with open(os.path.join(dataset_dir, dataset_name + '.zip'), 'rb') as data:
                self.s3.upload_fileobj(Fileobj=data, 
                                       Bucket='agdata-data',
                                       Key='datasets/' + dataset_name + '.zip',
                                       Callback=lambda x: self.pg.update(x))
        except:
            warnings.warn(
                f'Upload of {dataset_name} unsuccessful. You may not have permission '
                f'to upload to the agdata-data s3 bucket.', category = UserWarning)
        finally:
            self.pg.close()

    def upload_model(self, model_name, model_dir):
        """Uploads model to agdata-data s3 file storage.

        Parameters
        ----------
        model_name : str
            name of model (without '.pth')
        model_dir : str
            path to directory where model is stored
        """
        # Establish connection with s3 via boto
        self.s3 = boto3.client('s3')

        # Setup progress bar
        self.pg = tqdm(
            total=os.stat(os.path.abspath(os.path.join(model_dir, model_name + '.pth'))).st_size,
            file=sys.stdout, desc = f"Uploading {model_name}")

        # Upload data to agdata-data bucket
        try:
            with open(os.path.join(model_dir, model_name + '.pth'), 'rb') as data:
                self.s3.upload_fileobj(Fileobj=data, 
                                       Bucket='agdata-data',
                                       Key='models/' + model_name + '.pth',
                                       Callback=lambda x: self.pg.update(x))
        except:
            warnings.warn(
                f'Upload of {model_name} unsuccessful. You may not have permission '
                f'to upload to the agdata-data s3 bucket.', category = UserWarning)
        finally:
            self.pg.close()

    def download_dataset(self, dataset_name, dest_dir):
        """
        Downloads dataset from agdata-data s3 file storage.
        
        Parameters
        ----------
        dataset_name : str
            name of dataset to download
        dest_dir : str
            path for saving downloaded dataset
        """
        # Validate the dataset name.
        if dataset_name not in self.data_sources:
            raise ValueError(f"Invalid dataset '{dataset_name}.'")

        # Establish connection with s3 via boto
        self.s3 = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')

        # Setup progress bar
        try:
            self.pg = tqdm(
                total=float(self.s3_resource.ObjectSummary(
                    bucket_name='agdata-data', key=dataset_name + '.zip').size),
                file=sys.stdout, desc = f"Downloading {dataset_name}")
        except botocore.exceptions.ClientError as ce:
            if "Not Found" in str(ce):
                raise ValueError(
                    f"The dataset '{dataset_name}' could not be found in "
                    f"the bucket, perhaps it has not been uploaded yet.")
            raise ce

        # File path of zipped dataset
        self.dataset_download_path = os.path.join(dest_dir, dataset_name + '.zip')

        # Upload data to agdata-data bucket
        with open(self.dataset_download_path, 'wb') as data:
            self.s3.download_fileobj(Bucket='agdata-data', 
                                     Key=dataset_name + '.zip', 
                                     Fileobj=data,
                                     Callback=lambda x: self.pg.update(x))
        self.pg.close()

        # Unzip downloaded dataset
        with zipfile.ZipFile(self.dataset_download_path, 'r') as z:
            z.printdir()
            print('Extracting files...')
            z.extractall(path=dest_dir)
            print('Done!')

        # Delete zipped file
        os.remove(self.dataset_download_path)
