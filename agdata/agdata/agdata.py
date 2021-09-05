import json

class AgDataAPI:

    def __init__(self):
        self.data_srcs_path = 'assets/data_sources.json'
        with open(self.data_srcs_path) as f:
            self.data_srcs = json.load(f)