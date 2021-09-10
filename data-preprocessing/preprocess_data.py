import os
import csv

class PreprocessData:

    def __init__(self, data_dir, dataset_name):
        self.data_dir = data_dir
        self.data_original_dir = self.data_dir + 'original/'
        self.data_processed_dir = self.data_dir + 'processed/'

        self.dataset_name = None
    
    def pp_bean_disease_uganda(self):
        dataset_name = 'bean_disease_uganda'
    
    def pp_carrot_weeds_germany(self):
        dataset_name = 'carrot_weeds_germany'

    def pp_carrot_weeds_macedonia(self):
        dataset_name = 'carrot_weeds_macedonia'

    def pp_rangeland_weeds_australia(self):
        dataset_name = 'rangeland_weeds_australia'
        dataset_dir = self.data_original_dir + '/' + self.dataset_name + '/'
        labels = []
        labels_unique = []

        # Make directories with class names
        with open(dataset_dir) as f:
            labels.append(csv.reader(f))

        # Read through list, keep only unique classes, and create directories for each class name
        for label in labels:
            if label not in labels_unique:
                labels_unique.append(label)
            

        # Move files to folder with correct class names

        # dataset_dir = os.listdir(self.data_original_dir + '/' + self.dataset_name + '/' + labels)

