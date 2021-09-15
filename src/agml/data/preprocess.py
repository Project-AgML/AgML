import os
import sys
import json
import shutil

import numpy as np
from tqdm import tqdm

from .utils import get_filelist, get_dirlist, read_txt_file, get_image_info
from .utils import convert_txt_to_cocojson, get_label2id, create_dir, get_coco_annotation_from_obj

class PreprocessData:

    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.data_original_dir = os.path.join(self.data_dir, 'original')
        self.data_processed_dir = os.path.join(self.data_dir, 'processed')

    def preprocess(self, dataset_name):
        """Preprocesses the provided dataset.

        Parameters
        ----------
        dataset_name : str
            name of dataset to preprocess
        """
        if dataset_name == 'bean_disease_uganda':
            pass

        elif dataset_name == 'carrot_weeds_germany':
            pass

        elif dataset_name == 'carrot_weeds_macedonia':
            pass

        elif dataset_name == 'rangeland_weeds_australia':
            dataset_dir = os.path.join(self.data_original_dir, dataset_name)
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
                os.rename(imgs_dir + img_names[k], labels_dir + label + '/' + img_names[k])

        if dataset_name == 'fruits_classification_worldwide':
            dataset_dir = os.path.join(self.data_original_dir, dataset_name, 'datasets')

            # get folder list
            dataset_folders = get_dirlist(dataset_dir)
            label2id = get_label2id(dataset_folders)
            anno_data_all = []
            for folder in dataset_folders:
                annotations = ['test_RGB.txt', 'train_RGB.txt']
                dataset_path = os.path.join(dataset_dir, folder)
                # @TODO: Make separate json files for train and test?
                for anno_file_name in annotations:
                    # get img folder name
                    name = anno_file_name.split('.')[0].upper()

                    # Read annotations
                    try:
                        anno_data = read_txt_file(os.path.join(dataset_path, anno_file_name))
                    except:
                        try:
                            anno_data = read_txt_file(os.path.join(dataset_path, anno_file_name + '.txt'))
                        except:
                            raise

                    # Concat fruit name at head of line
                    for i, anno in enumerate(anno_data):
                        # Change to test path if the text file is test
                        if "test" in anno_file_name and "TRAIN" in anno[0]:
                            anno_data[i][0] = anno[0].replace("TRAIN", "TEST")
                        anno_data[i][0] = os.path.join(dataset_path, anno_data[i][0])

                    anno_data_all += anno_data

            # Process annotation files
            save_dir_anno = os.path.join(self.data_processed_dir, dataset_name, 'annotations')
            create_dir(save_dir_anno)
            output_json_file = os.path.join(save_dir_anno, 'train.json')

            general_info = {
                "description": "fruits dataset",
                "url": "https://drive.google.com/drive/folders/1CmsZb1caggLRN7ANfika8WuPiywo4mBb",
                "version": "1.0",
                "year": 2018,
                "contributor": "Inkyu Sa",
                "date_created": "2018/11/12"
            }

            convert_txt_to_cocojson(
                anno_data_all, label2id, output_json_file, general_info)

            # Process image files
            save_dir_imgs = os.path.join(self.data_processed_dir, dataset_name, 'images')
            create_dir(save_dir_imgs)
            for anno in tqdm(anno_data_all):
                img_name = anno[0].split('/')[-1]
                dest_path = os.path.join(save_dir_imgs, img_name)
                try:
                    shutil.copyfile(anno[0], dest_path)
                except:
                    # Cannot copy the image file
                    pass

        elif dataset_name == 'cotton_seedling_counting':
            # Get all of the relevant data
            dataset_dir = os.path.join(self.data_original_dir, dataset_name)
            image_dir = os.path.join(dataset_dir, 'Images')
            images = sorted([os.path.join(image_dir, i) for i in os.listdir(image_dir)])
            with open(os.path.join(dataset_dir, 'Images.json'), 'r') as f:
                annotations = json.load(f)

            # Get all of the unique labels
            labels = []
            for label_set in annotations['frames'].values():
                for individual_set in label_set:
                    labels.extend(individual_set['tags'])
            labels = np.unique(labels).tolist()
            label2id = get_label2id(labels) # noqa

            # Extract all of the bounding boxes and images
            image_data = []
            annotation_data = []
            valid_paths = [] # some paths are not in the annotations, track the ones which are
            for indx, (img_path, annotation) in enumerate(
                    zip(tqdm(images, file = sys.stdout, desc = "Generating Data"),
                        annotations['frames'].values())):
                image_data.append(get_image_info(img_path, indx))
                valid_paths.append(img_path)
                for a_set in annotation:
                    formatted_set = [
                        a_set['x1'], a_set['y1'], a_set['x2'], a_set['y2'],
                        label2id[a_set['tags'][0]]]
                    base_annotation_data = get_coco_annotation_from_obj(formatted_set, a_set['name'])
                    base_annotation_data['image_id'] = indx + 1
                    annotation_data.append(base_annotation_data)

            # Set up the annotation dictionary
            all_annotation_data = {
                "images": [], "type": "instances",
                "annotations": [], "categories": [],
                "info": {
                    "description": "cotton seedling counting dataset",
                    "url": "https://figshare.com/s/616956f8633c17ceae9b",
                    "version": "1.0",
                    "year": 2019,
                    "contributor": "Yu Jiang",
                    "date_created": "2019/11/23"
                }
            }

            # Populate the annotation dictionary
            for label, label_id in label2id.items():
                category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
                all_annotation_data['categories'].append(category_info)
            all_annotation_data['images'] = image_data
            all_annotation_data['annotations'] = annotation_data

            # Recreate the dataset and zip it
            processed_dir = os.path.join(self.data_processed_dir, dataset_name)
            processed_img_dir = os.path.join(processed_dir, 'images')
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir)
            os.makedirs(processed_dir, exist_ok = True)
            os.makedirs(processed_img_dir, exist_ok = True)
            for path in images:
                if path not in valid_paths:
                    continue
                shutil.copyfile(path, os.path.join(processed_img_dir, os.path.basename(path)))
            with open(os.path.join(processed_dir, 'labels.json'), 'w') as f:
                json.dump(all_annotation_data, f, indent = 4)

            # Zip the dataset
            shutil.make_archive(
                processed_dir, "zip", os.path.dirname(processed_dir))


