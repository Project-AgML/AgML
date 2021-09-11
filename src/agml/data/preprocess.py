import os
import json

class DataPreprocessor(object):
    """
    Preprocesses an AgML dataset.

    Attributes
    ----------
    No user specified attributes.
    """
    def __init__(self, data_dir):
        # Set the default data directories
        self.data_dir = data_dir
        self.data_original_dir = os.path.join(self.data_dir, 'original')
        self.data_processed_dir = os.path.join(self.data_dir, 'processed')

        # Get a list of all of the potential datasets
        with open(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'assets', 'public_datasources.json')) as f:
            self.data_srcs = json.load(f)

    def preprocess(self, dataset_name):
        """Preprocesses a dataset.

        dataset_name : str
            The name of the dataset to process.
        """
        # Validate the dataset name
        if dataset_name not in self.data_srcs:
            raise ValueError(f"Invalid dataset name {dataset_name}.")

        if dataset_name == 'bean_disease_uganda':
            pass

        elif dataset_name == 'carrot_weeds_germany':
            pass

        elif dataset_name == 'carrot_weeds_macedonia':
            pass

        elif dataset_name == 'rangeland_weeds_australia':
            dataset_dir = os.path.join(
                self.data_original_dir, 'rangeland_weeds_australia')
            imgs_dir = os.path.join(dataset_dir, 'images')
            labels_dir = os.path.join(dataset_dir, 'labels')
            labels_path = os.path.join(labels_dir, 'labels.csv')
            labels_unique = []

            # Make directories with class names
            with open(labels_path) as f:
                next(f)
                labels = [row.split(',')[2] for row in f]

            with open(labels_path) as f:
                next(f)
                img_names = [row.split(',')[0].strip().replace(' ', '_') for row in f]

            # Read through list, keep only unique classes, and create directories for each class name
            for k, label in enumerate(labels):
                if label not in labels_unique:
                    labels_unique.append(label)
                    os.mkdir(labels_dir + label)
                os.rename(imgs_dir + img_names[k], os.path.join(labels_dir, label, img_names[k]))

