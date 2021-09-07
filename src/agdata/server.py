import json
import os
import sys
import boto3
import progressbar

class AgDataServer:

    def __init__(self):

        # Read in meta-data for data sources file
        self.data_srcs_path = 'assets/data_sources.json'
        with open(self.data_srcs_path) as f:
            self.data_srcs = json.load(f)

        # Define s3 bucket URI
        self.agdata_s3_uri = 'https://s3.us-west-1.amazonaws.com/agdata-data/'


    def upload_dataset(self, dataset_dir, dataset_name):

        # Establish connection with s3 via boto
        self.s3 = boto3.client('s3')

        # Setup progress bar
        self.pg = progressbar.progressbar.ProgressBar(
            maxval=os.stat(dataset_dir + '/' + dataset_name + '.zip').st_size)
        self.pg.start()

        # Upload data to agdata-data bucket
        with open(dataset_dir + '/' + dataset_name + '.zip', 'rb') as data:
            self.s3.upload_fileobj(data, 
                                   bucket='agdata-data', 
                                   key=dataset_name + '.zip', 
                                   Callback=self.updownload_callback)   


    def download_dataset(self, dest_dir, dataset_name):

        # Establish connection with s3 via boto
        self.s3 = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')

        # Setup progress bar
        self._size = float(self.s3_resource.ObjectSummary(bucket_name='agdata-data', key=dataset_name + '.zip').size)
        self.pg = progressbar.progressbar.ProgressBar(maxval=self._size)
        self.pg.start()

        # Upload data to agdata-data bucket
        with open(dest_dir + '/' + dataset_name + '.zip', 'wb') as data:
            self.s3.download_fileobj(Bucket='agdata-data', 
                                     Key=dataset_name + '.zip', 
                                     Fileobj=data,
                                     Callback=self.updownload_callback)  


    def updownload_callback(self, size):
        self.pg.update(self.pg.currval + size)