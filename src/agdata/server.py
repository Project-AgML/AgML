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


    def put_data(self, dataset_dir, dataset_name):

        # Upload the zipped dataset file
        # s3 = boto3.resource('s3')
        # data = open(dataset_dir + '/' + dataset_name + '.zip', 'rb')
        # s3.Bucket('agdata-data').put_object(Key=dataset_name + '.zip', Body=data)

        self.s3 = boto3.client('s3')
        self.upload(bucket='agdata-data',
                    key=dataset_name + '.zip',
                    file=dataset_dir + '/' + dataset_name + '.zip')

    def upload_callback(self, size):
        self.pg.update(self.pg.currval + size)

    def upload(self, bucket, key, file):
        self.pg = progressbar.progressbar.ProgressBar(
            maxval=os.stat(file).st_size)
        self.pg.start()

        with open(file, 'rb') as data:
            self.s3.upload_fileobj(
                data, bucket, key, Callback=self.upload_callback)