import json
import urllib
import boto3

class AgDataAPI:

    def __init__(self):

        # Read in meta-data for data sources file
        self.data_srcs_path = 'assets/data_sources.json'
        with open(self.data_srcs_path) as f:
            self.data_srcs = json.load(f)

        # Define s3 bucket URI
        self.agdata_s3_uri = 'https://s3.us-west-1.amazonaws.com/agdata-data/'

    def download_dataset(self, dataset_name):
        url = self.data_srcs['datasets'][dataset_name]['data_url']