import json
import os
import sys
import zipfile
import boto3
import progressbar

class PublicDataAPI: 

    """
    A class for working with AgData

    ...
    Attributes
    ----------
    No user specified attributes

    """

    def __init__(self):

        # Read in meta-data for data sources file
        self.data_srcs_path = 'assets/public_datasources.json'
        with open(self.data_srcs_path) as f:
            self.data_srcs = json.load(f)

        # Define s3 bucket URI
        self.agdata_s3_uri = 'https://s3.us-west-1.amazonaws.com/agdata-data/'

        # Initialize attribute for storing dataset download path
        self.dataset_download_path = None


    def upload_dataset(self, dataset_dir, dataset_name):

        """
        Uploads dataset to agdata-data s3 file storage
        
        Parameters
        ----------
            dataset_dir : str
                path to directory where dataset is stored
            dataset_name : str
                name of dataset (without '.zip') -- for list of datasets run self.data_srcs.keys()
        """

        # Establish connection with s3 via boto
        self.s3 = boto3.client('s3')

        # Setup progress bar
        self.pg = progressbar.progressbar.ProgressBar(
            maxval=os.stat(dataset_dir + '/' + dataset_name + '.zip').st_size)
        self.pg.start()

        # Upload data to agdata-data bucket
        try:
            with open(dataset_dir + '/' + dataset_name + '.zip', 'rb') as data:
                self.s3.upload_fileobj(Fileobj=data, 
                                    Bucket='agdata-data', 
                                    Key=dataset_name + '.zip', 
                                    Callback=self.updownload_callback)
        except:
            print('Upload unsuccessful. You may not have permission to upload to the agdata-data s3 bucket...')


    def download_dataset(self, dataset_name, dest_dir):

        """
        Downloads dataset from agdata-data s3 file storage
        
        Parameters
        ----------
            dataset_name : str
                name of dataset to download
            dest_dir : str
                path for saving downloaded dataset
        """

        # Establish connection with s3 via boto
        self.s3 = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')

        # Setup progress bar
        self._size = float(self.s3_resource.ObjectSummary(bucket_name='agdata-data', key=dataset_name + '.zip').size)
        self.pg = progressbar.progressbar.ProgressBar(maxval=self._size)
        self.pg.start()

        # File path of zipped dataset
        self.dataset_download_path = dest_dir + '/' + dataset_name + '.zip'

        # Upload data to agdata-data bucket
        with open(self.dataset_download_path, 'wb') as data:
            self.s3.download_fileobj(Bucket='agdata-data', 
                                     Key=dataset_name + '.zip', 
                                     Fileobj=data,
                                     Callback=self.updownload_callback)

        # Unzip downloaded dataset
        with zipfile.ZipFile(self.dataset_download_path, 'r') as zip:
            zip.printdir()
            print('Extracting files...')
            zip.extractall(path=dest_dir)
            print('Done!')

        # Delete zipped file
        os.remove(self.dataset_download_path)


    def updownload_callback(self, size):

        """
        Callback function for tracking download or upload progress
        """
        
        self.pg.update(self.pg.currval + size)