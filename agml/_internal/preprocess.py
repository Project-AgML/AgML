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
Preprocessing code for AgML public data sources.
This file stores the preprocessing code used to preprocess a public
dataset when added to AgML's public data sources.
If you want to use this preprocessing code, run `pip install agml[dev]`
to install the necessary preprocessing packages.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from agml._internal.process_utils import (
    convert_bbox_to_coco,
    convert_xmls_to_cocojson,
    get_coco_annotation_from_obj,
    get_image_info,
    get_label2id,
    move_segmentation_dataset,
    read_txt_file,
)
from agml.utils.data import load_public_sources
from agml.utils.io import create_dir, get_dir_list, get_file_list, nested_dir_list
from agml.utils.logging import tqdm

def yolo_to_coco_bbox(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format (x_center, y_center, width, height) to COCO format (xmin, ymin, width, height)."""
    xmin = (x_center - (width / 2)) * img_width
    ymin = (y_center - (height / 2)) * img_height
    width = width * img_width
    height = height * img_height
    return [xmin, ymin, width, height]


class PublicDataPreprocessor(object):
    """Internal data preprocessing class.

    Parameters
    ----------
    data_dir : str
        The directory with a folder `original` and `processed` to hold
        the original and processed datasets, respectively.
    """

    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.data_original_dir = os.path.join(self.data_dir, "original")
        self.data_processed_dir = os.path.join(self.data_dir, "processed")
        self.data_sources = load_public_sources()

    def preprocess(self, dataset_name):
        """Preprocesses the provided dataset.
        Parameters
        ----------
        dataset_name : str
            name of dataset to preprocess
        """
        getattr(self, dataset_name)(dataset_name)

    def vine_virus_photo_dataset(self, dataset_name):
        """Preprocesses the Vine Virus Photo Dataset."""
        # Get the dataset directory directly (no need for 'original' directory)
        base_path = self.data_dir
        classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

        # Create output directory
        output_path = os.path.join(self.data_processed_dir, dataset_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Create subdirectories for each class inside the output directory
        for cls in classes:
            class_output_path = os.path.join(output_path, cls)
            if not os.path.exists(class_output_path):
                os.makedirs(class_output_path)

        # Process and copy the dataset images to the processed directory
        for cls in classes:
            class_path = os.path.join(base_path, cls)
            for img in os.listdir(class_path):
                if img.endswith(("jpg", "png", "jpeg", "JPG")):
                    img_path = os.path.join(class_path, img)
                    shutil.copyfile(img_path, os.path.join(output_path, cls, img))

        print(f"Dataset {dataset_name} has been preprocessed and saved to {output_path}")

    def corn_maize_leaf_disease(self, dataset_name):
        """Preprocesses the Corn or Maize Leaf Disease Dataset."""
        # Get the dataset directory directly (no need for 'original' directory)
        base_path = self.data_dir
        classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

        # Create output directory
        output_path = os.path.join(self.data_processed_dir, dataset_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Create subdirectories for each class inside the output directory
        for cls in classes:
            class_output_path = os.path.join(output_path, cls)
            if not os.path.exists(class_output_path):
                os.makedirs(class_output_path)

        # Process and copy the dataset images to the processed directory
        for cls in classes:
            class_path = os.path.join(base_path, cls)
            for img in os.listdir(class_path):
                if img.endswith(("jpg", "png", "jpeg", "JPG")):
                    img_path = os.path.join(class_path, img)

                    # Open the image using Pillow
                    with Image.open(img_path) as image:
                        # Convert image to a NumPy array
                        img_array = np.array(image)

                        # Ensure the image is in range [0, 255] and dtype is uint8
                        if img_array.dtype != np.uint8:
                            # If the image is in float [0-1], scale it to [0-255]
                            img_array = (img_array * 255).astype(np.uint8)

                        # Convert back to PIL Image to save
                        processed_image = Image.fromarray(img_array)

                        if processed_image.mode == "RGBA":
                            processed_image = processed_image.convert("RGB")

                        # Save the processed image to the output directory
                        processed_image.save(os.path.join(output_path, cls, img))

        print(f"Dataset {dataset_name} has been preprocessed and saved to {output_path}")

    def tomato_leaf_disease(self, dataset_name):
        """Preprocesses the Tomato Leaf Disease Dataset."""
        # Get the dataset directory directly (no need for 'original' directory)
        base_path = self.data_dir
        classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

        # Create output directory
        output_path = os.path.join(self.data_processed_dir, dataset_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Create subdirectories for each class inside the output directory
        for cls in classes:
            class_output_path = os.path.join(output_path, cls)
            if not os.path.exists(class_output_path):
                os.makedirs(class_output_path)

        # Process and copy the dataset images to the processed directory
        for cls in classes:
            class_path = os.path.join(base_path, cls)
            for img in os.listdir(class_path):
                if img.endswith(("jpg", "png", "jpeg", "JPG")):
                    img_path = os.path.join(class_path, img)

                    # Open the image using Pillow
                    with Image.open(img_path) as image:
                        # Convert image to a NumPy array
                        img_array = np.array(image)

                        # Ensure the image is in range [0, 255] and dtype is uint8
                        if img_array.dtype != np.uint8:
                            # If the image is in float [0-1], scale it to [0-255]
                            img_array = (img_array * 255).astype(np.uint8)

                        # Convert back to PIL Image to save
                        processed_image = Image.fromarray(img_array)

                        if processed_image.mode == "RGBA":
                            processed_image = processed_image.convert("RGB")

                        # Save the processed image to the output directory
                        processed_image.save(os.path.join(output_path, cls, img))

        print(f"Dataset {dataset_name} has been preprocessed and saved to {output_path}")

    def bean_disease_uganda(self, dataset_name):
        # Get the dataset classes and paths
        base_path = os.path.join(self.data_original_dir, dataset_name)
        dirs = ["train", "validation", "test"]
        classes = sorted(os.listdir(os.path.join(base_path, dirs[0])))[1:]

        # Construct output directories
        output = os.path.join(self.data_processed_dir, dataset_name)
        os.makedirs(output, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(output, cls), exist_ok=True)

        # Move the dataset
        for dir_ in dirs:
            for cls in classes:
                path = os.path.join(base_path, dir_, cls)
                for p in os.listdir(path):
                    if p.endswith("jpg") or p.endswith("png"):
                        img = os.path.join(base_path, dir_, cls, p)
                        shutil.copyfile(img, os.path.join(output, cls, p))

    def leaf_counting_denmark(self, dataset_name):
        pass

    def plant_seedlings_aarhus(self, dataset_name):
        pass

    def crop_weeds_greece(self, dataset_name):
        pass

    def rangeland_weeds_australia(self, dataset_name):
        # Get the file information.
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        images = get_file_list(os.path.join(dataset_dir, "images"))
        df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))

        # Construct the new structure.
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        unique_labels = np.unique(df["Species"])
        for unique_label in unique_labels:
            os.makedirs(os.path.join(processed_dir, unique_label.title()), exist_ok=True)
        for file in tqdm(images, desc="Moving Images", file=sys.stdout):
            save_dir = df.loc[df["Filename"] == os.path.basename(file)]["Species"].values[0].title()
            shutil.copyfile(
                os.path.join(dataset_dir, "images", file),
                os.path.join(processed_dir, save_dir, os.path.basename(file)),
            )

    def fruit_detection_worldwide(self, dataset_name):
        # Get the dataset directory
        dataset_dir = os.path.join(self.data_original_dir, dataset_name, "datasets")

        # Get folder list
        dataset_folders = get_dir_list(dataset_dir)
        label2id = get_label2id(dataset_folders)
        anno_data_all = []
        for folder in dataset_folders:
            annotations = ["test_RGB.txt", "train_RGB.txt"]
            dataset_path = os.path.join(dataset_dir, folder)
            # @TODO: Make separate json files for train and test?
            for anno_file_name in annotations:
                # Read annotations
                try:
                    anno_data = read_txt_file(os.path.join(dataset_path, anno_file_name))
                except:
                    try:
                        anno_data = read_txt_file(os.path.join(dataset_path, anno_file_name + ".txt"))
                    except Exception as e:
                        raise e

                # Concat fruit name at head of line
                for i, anno in enumerate(anno_data):
                    # Change to test path if the text file is test
                    if "test" in anno_file_name and "TRAIN" in anno[0]:
                        anno_data[i][0] = anno[0].replace("TRAIN", "TEST")
                    anno_data[i][0] = os.path.join(dataset_path, anno_data[i][0])

                anno_data_all += anno_data

        # Process annotation files
        save_dir_anno = os.path.join(self.data_processed_dir, dataset_name, "annotations")
        create_dir(save_dir_anno)
        output_json_file = os.path.join(save_dir_anno, "instances.json")

        general_info = {
            "description": "fruits dataset",
            "url": "https://drive.google.com/drive/folders/1CmsZb1caggLRN7ANfika8WuPiywo4mBb",
            "version": "1.0",
            "year": 2018,
            "contributor": "Inkyu Sa",
            "date_created": "2018/11/12",
        }

        # Process image files
        output_img_path = os.path.join(self.data_processed_dir, dataset_name, "images")
        create_dir(output_img_path)

        convert_bbox_to_coco(anno_data_all, label2id, output_json_file, output_img_path, general_info)

    def apple_detection_usa(self, dataset_name, fix=False):
        # Just a quick fix to clip over-sized bounding boxes.
        if fix:
            # Load in the annotations.
            dataset_dir = os.path.join(self.data_original_dir, dataset_name)
            with open(os.path.join(dataset_dir, "annotations.json"), "r") as f:
                annotations = json.load(f)

            # Get the images and all of their heights/widths.
            images = annotations["images"]
            image_id_content_map = {}
            for image in images:
                image_id_content_map[image["id"]] = (image["height"], image["width"])

            # Load all of the annotations.
            new_annotations = []
            for a in annotations["annotations"]:
                new_a = a.copy()
                height, width = image_id_content_map[a["image_id"]]
                (x, y, w, h) = a["bbox"]
                x1, y1, x2, y2 = x, y, x + w, y + h
                x1 = np.clip(x1, 0, width)
                x2 = np.clip(x2, 0, width)
                y1 = np.clip(y1, 0, height)
                y2 = np.clip(y2, 0, height)
                new_a["bbox"] = [int(i) for i in [x1, y1, x2 - x1, y2 - y1]]
                new_annotations.append(new_a)

            # Save the annotations.
            annotations["annotations"] = new_annotations
            with open(os.path.join(dataset_dir, "annotations.json"), "w") as f:
                json.dump(annotations, f)
            return

        # resize the dataset
        resize = 1.0

        # Read public_datasources.json to get class information
        category_info = self.data_sources[dataset_name]["classes"]
        labels_str = []
        labels_ids = []
        for info in category_info:
            labels_str.append(category_info[info])
            labels_ids.append(int(info))

        label2id = dict(zip(labels_str, labels_ids))

        # Task 1: Image classification
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        obj_Detection_data = os.path.join(dataset_dir, "Dataset")

        # get folders
        plant_folders = nested_dir_list(obj_Detection_data)

        # do tasks along folders
        anno_data_all = []
        for folder in plant_folders:
            # Get image file and xml file
            full_path = os.path.join(obj_Detection_data, folder)
            all_files = get_file_list(full_path)
            anno_files = [x for x in all_files if "txt" in x]
            for anno_file in anno_files:
                anno_line = []
                anno_path = os.path.join(full_path, anno_file)
                # Opening annotation file
                anno_data = read_txt_file(anno_path, delimiter=",")[0]

                for i, anno in enumerate(anno_data):
                    new_anno = [os.path.join(dataset_dir, anno_data[i][0])]
                    # Add bbox count
                    # Update image file path to abs path
                    bbox_cnt = int((len(anno_data[i]) - 1) / 4)
                    new_anno.append(str(bbox_cnt))
                    for idx in range(bbox_cnt):
                        xmin = int(anno[1 + 4 * idx])
                        ymin = int(anno[1 + 4 * idx + 1])
                        w = int(anno[1 + 4 * idx + 2])
                        h = int(anno[1 + 4 * idx + 3])

                        new_anno.append(str(xmin))  # xmin
                        new_anno.append(str(ymin))  # ymin
                        new_anno.append(str(xmin + w))  # xmax
                        new_anno.append(str(ymin + h))  # ymax
                        new_anno.append(str(1))  # label
                    anno_data[i] = new_anno
                anno_data_all += anno_data

        # Process annotation files
        save_dir_anno = os.path.join(self.data_processed_dir, dataset_name, "annotations")
        create_dir(save_dir_anno)
        output_json_file = os.path.join(save_dir_anno, "instances.json")

        general_info = {
            "description": "apple dataset",
            "url": "https://research.libraries.wsu.edu:8443/xmlui/handle/2376/17721",
            "version": "1.0",
            "year": 2019,
            "contributor": "Bhusal, Santosh, Karkee, Manoj, Zhang, Qin",
            "date_created": "2019/04/20",
        }

        # Process image files
        output_img_path = os.path.join(self.data_processed_dir, dataset_name, "images")
        create_dir(output_img_path)
        convert_bbox_to_coco(
            anno_data_all,
            label2id,
            output_json_file,
            output_img_path,
            general_info,
            None,
            None,
            get_label_from_folder=False,
            resize=resize,
            add_foldername=True,
        )

    def mango_detection_australia(self, dataset_name):
        # resize the dataset
        resize = 1.0

        # Read public_datasources.json to get class information
        datasource_file = os.path.join(os.path.dirname(__file__), "../_assets/public_datasources.json")
        with open(datasource_file) as f:
            data = json.load(f)
            category_info = data[dataset_name]["crop_types"]
            labels_str = []
            labels_ids = []
            for info in category_info:
                labels_str.append(category_info[info])
                labels_ids.append(int(info))

            name_converter = dict(zip(["M"], ["mango"]))  # src -> dst
            label2id = dict(zip(labels_str, labels_ids))

        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        ann_dir = os.path.join(dataset_dir, "VOCDevkit/VOC2007/Annotations")

        # Get image file and xml file
        all_files = get_file_list(ann_dir)
        anno_files = [os.path.join(ann_dir, x) for x in all_files if "xml" in x]
        img_files = [x.replace(".xml", ".jpg").replace("Annotations", "JPEGImages") for x in anno_files]

        # Process annotation files
        save_dir_anno = os.path.join(self.data_processed_dir, dataset_name, "annotations")
        create_dir(save_dir_anno)
        output_json_file = os.path.join(save_dir_anno, "instances.json")

        # Process image files
        output_img_path = os.path.join(self.data_processed_dir, dataset_name, "images")
        create_dir(output_img_path)

        general_info = {
            "description": "MangoYOLO data set",
            "url": "https://researchdata.edu.au/mangoyolo-set/1697505",
            "version": "1.0",
            "year": 2019,
            "contributor": "Anand Koirala, Kerry Walsh, Z Wang, C McCarthy",
            "date_created": "2019/02/25",
        }

        convert_xmls_to_cocojson(
            general_info,
            annotation_paths=anno_files,
            img_paths=img_files,
            label2id=label2id,
            name_converter=name_converter,
            output_jsonpath=output_json_file,
            output_imgpath=output_img_path,
            extract_num_from_imgid=True,
        )

    def tomato_ripeness_detection(self, dataset_name):
        pass

    def cotton_seedling_counting(self, dataset_name):
        # Get all of the relevant data
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        image_dir = os.path.join(dataset_dir, "Images")
        images = sorted([os.path.join(image_dir, i) for i in os.listdir(image_dir)])
        with open(os.path.join(dataset_dir, "Images.json"), "r") as f:
            annotations = json.load(f)

        # Get all of the unique labels
        labels = []
        for label_set in annotations["frames"].values():
            for individual_set in label_set:
                labels.extend(individual_set["tags"])
        labels = np.unique(labels).tolist()
        label2id = get_label2id(labels)  # noqa

        # Extract all of the bounding boxes and images
        image_data = []
        annotation_data = []
        valid_paths = []  # some paths are not in the annotations, track the ones which are
        for indx, (img_path, annotation) in enumerate(
            zip(
                tqdm(images, file=sys.stdout, desc="Generating Data"),
                annotations["frames"].values(),
            )
        ):
            image_data.append(get_image_info(img_path, indx))
            valid_paths.append(img_path)
            for a_set in annotation:
                formatted_set = [
                    a_set["x1"],
                    a_set["y1"],
                    a_set["x2"],
                    a_set["y2"],
                    label2id[a_set["tags"][0]],
                ]
                base_annotation_data = get_coco_annotation_from_obj(formatted_set, a_set["name"])
                base_annotation_data["image_id"] = indx + 1
                annotation_data.append(base_annotation_data)

        # Set up the annotation dictionary
        all_annotation_data = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": [],
            "info": {
                "description": "cotton seedling counting dataset",
                "url": "https://figshare.com/s/616956f8633c17ceae9b",
                "version": "1.0",
                "year": 2019,
                "contributor": "Yu Jiang",
                "date_created": "2019/11/23",
            },
        }

        # Populate the annotation dictionary
        for label, label_id in label2id.items():
            category_info = {"supercategory": "none", "id": label_id, "name": label}
            all_annotation_data["categories"].append(category_info)
        all_annotation_data["images"] = image_data
        all_annotation_data["annotations"] = annotation_data

        # Recreate the dataset and zip it
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_img_dir = os.path.join(processed_dir, "images")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(processed_img_dir, exist_ok=True)
        for path in images:
            if path not in valid_paths:
                continue
            shutil.copyfile(path, os.path.join(processed_img_dir, os.path.basename(path)))
        with open(os.path.join(processed_dir, "annotations.json"), "w") as f:
            json.dump(all_annotation_data, f, indent=4)

        # Zip the dataset
        shutil.make_archive(processed_dir, "zip", os.path.dirname(processed_dir))

    def apple_flower_segmentation(self, dataset_name):
        # Get all of the relevant data.
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        apple_a_dir = os.path.join(dataset_dir, "FlowerImages")
        apple_a_images = os.listdir(apple_a_dir)
        apple_a_label_dir = os.path.join(dataset_dir, "AppleA_Labels")
        apple_a_labels = os.listdir(apple_a_label_dir)
        apple_b_dir = os.path.join(dataset_dir, "AppleB")
        apple_b_images = os.listdir(apple_b_dir)
        apple_b_label_dir = os.path.join(dataset_dir, "AppleB_Labels")
        apple_b_labels = os.listdir(apple_b_label_dir)

        # Map image filenames with their corresponding labels.
        fname_map_a, fname_map_b = {}, {}
        for fname in apple_a_images:
            fname_id = str(int(float(os.path.splitext(fname)[0].split("_")[-1]))) + ".png"
            if fname_id in apple_a_labels:
                fname_map_a[os.path.join(apple_a_dir, fname)] = os.path.join(apple_a_label_dir, fname_id)
        for fname in apple_b_images:
            fname_id = str(int(float(os.path.splitext(fname)[0].split("_")[-1]))) + ".png"
            if fname_id in apple_b_labels:
                fname_map_b[os.path.join(apple_b_dir, fname)] = os.path.join(apple_b_label_dir, fname_id)

        # Process and move the images.
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        os.makedirs(processed_dir, exist_ok=True)
        processed_image_dir = os.path.join(processed_dir, "images")
        os.makedirs(processed_image_dir, exist_ok=True)
        processed_annotation_dir = os.path.join(processed_dir, "annotations")
        os.makedirs(processed_annotation_dir, exist_ok=True)
        for image_path, label_path in tqdm(fname_map_a.items(), desc="Processing Part A", file=sys.stdout):
            image = cv2.resize(cv2.imread(image_path), (2074, 1382))
            label = cv2.resize(cv2.imread(label_path), (2074, 1382)) // 255
            label_path = os.path.basename(label_path)
            out_image_path = os.path.join(processed_image_dir, label_path)
            out_label_path = os.path.join(processed_annotation_dir, label_path)
            cv2.imwrite(out_image_path.replace(".png", ".jpg"), image)
            cv2.imwrite(out_label_path, label)
        for image_path, label_path in tqdm(fname_map_b.items(), desc="Processing Part B", file=sys.stdout):
            image = cv2.resize(cv2.imread(image_path), (2074, 1382))
            label = cv2.resize(cv2.imread(label_path), (2074, 1382)) // 255
            label_path = os.path.basename(label_path)
            out_image_path = os.path.join(processed_image_dir, label_path)
            out_label_path = os.path.join(processed_annotation_dir, label_path)
            cv2.imwrite(out_image_path.replace(".png", ".jpg"), image)
            cv2.imwrite(out_label_path, label)

    def sugarbeet_weed_segmentation(self, dataset_name):
        # Get all of the relevant data
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        train_dir = os.path.join(dataset_dir, "train")
        train_images = sorted(get_file_list(train_dir))
        annotation_dir = os.path.join(dataset_dir, "trainannot")  # noqa
        annotation_images = sorted(get_file_list(annotation_dir))

        # Move the images to the new directory
        move_segmentation_dataset(
            self.data_processed_dir,
            dataset_name,
            train_images,
            annotation_images,
            train_dir,
            annotation_dir,
        )

    def carrot_weeds_germany(self, dataset_name):
        # Get all of the relevant data.
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        train_dir = os.path.join(dataset_dir, "images")
        train_images = sorted(get_file_list(train_dir))
        annotation_dir = os.path.join(dataset_dir, "annotations")
        annotation_images = sorted(get_file_list(annotation_dir, ext="png"))

        # Move the images to the new directory.
        def _annotation_preprocess_fn(annotation_path, out_path):
            an_img = cv2.cvtColor(cv2.imread(annotation_path), cv2.COLOR_BGR2RGB)
            crop, weed = (0, 255, 0), (255, 0, 0)
            out_annotation = np.zeros(shape=an_img.shape[:-1])
            crop_indices = np.stack(np.where(np.all(an_img == crop, axis=-1))).T
            weed_indices = np.stack(np.where(np.all(an_img == weed, axis=-1))).T
            for indxs in crop_indices:
                out_annotation[indxs[0]][indxs[1]] = 1
            for indxs in weed_indices:
                out_annotation[indxs[0]][indxs[1]] = 2
            return cv2.imwrite(out_path, out_annotation.astype(np.int8))

        move_segmentation_dataset(
            self.data_processed_dir,
            dataset_name,
            train_images,
            annotation_images,
            train_dir,
            annotation_dir,
            annotation_preprocess_fn=_annotation_preprocess_fn,
        )

    def apple_segmentation_minnesota(self, dataset_name):
        # Get all of the relevant data.
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        train_dir = os.path.join(dataset_dir, "train", "images")
        train_images = sorted(get_file_list(train_dir))
        masks_dir = os.path.join(dataset_dir, "train", "masks")
        mask_images = sorted(get_file_list(masks_dir))

        # Move the images to the new directory.
        def _annotation_preprocess_fn(annotation_path, out_path):
            mask = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            ids = np.unique(mask)[1:]
            masks = mask == ids[:, np.newaxis, np.newaxis]
            masks = masks.astype(np.int32)
            if len(masks) == 1:
                mask = mask
            elif len(masks) >= 2:
                mask = np.logical_or(masks[0], masks[1])
                if len(masks) > 2:
                    for mask_ in masks[2:]:
                        mask = np.logical_or(mask, mask_)
            mask = mask.astype(np.int32)
            return cv2.imwrite(out_path, mask)

        move_segmentation_dataset(
            self.data_processed_dir,
            dataset_name,
            train_images,
            mask_images,
            train_dir,
            masks_dir,
            annotation_preprocess_fn=_annotation_preprocess_fn,
        )

    def rice_seedling_segmentation(self, dataset_name, fix=False):
        # Re-mapping labels to remove the `Background` class.
        if fix:
            data_dir = os.path.join(self.data_original_dir, dataset_name)
            annotations = sorted(
                [os.path.join(data_dir, "annotations", i) for i in os.listdir(os.path.join(data_dir, "annotations"))]
            )
            os.makedirs(os.path.join(data_dir, "new_annotations"))

            # Create the remap.
            for annotation in tqdm(annotations):
                a = cv2.imread(annotation)
                a[a == 2] = 0
                a[a == 3] = 2
                cv2.imwrite(
                    os.path.join(data_dir, "new_annotations", os.path.basename(annotation)),
                    a,
                )
            return

        # Get all of the relevant data.
        data_dir = os.path.join(self.data_original_dir, dataset_name)
        images = sorted(glob.glob(os.path.join(data_dir, "image_*.jpg")))
        labels = sorted(glob.glob(os.path.join(data_dir, "Label_*.png")))
        images = [os.path.basename(p) for p in images]
        labels = [os.path.basename(p) for p in labels]

        # Move the images to the new directory.
        move_segmentation_dataset(self.data_processed_dir, dataset_name, images, labels, data_dir, data_dir)

    def sugarcane_damage_usa(self, dataset_name):
        pass

    def soybean_weed_uav_brazil(self, dataset_name):
        pass

    def plant_village_classification(self, dataset_name):
        pass

    def autonomous_greenhouse_regression(self, dataset_name):
        # Get all of the data paths.
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        image_dir = os.path.join(dataset_dir, "RGBImages")
        depth_dir = os.path.join(dataset_dir, "DepthImages")
        with open(os.path.join(dataset_dir, "dataset.json"), "r") as f:
            contents = json.load(f)

        # Construct the output annotation JSON file.
        out = []
        for sample in contents["labels"].values():
            out.append(
                {
                    "image": sample["rgb_image_path"],
                    "depth_image": sample["depth_image_path"],
                    "outputs": {
                        "regression": sample["regression_outputs"],
                        "classification": sample["classification_outputs"]["Variety"],
                    },
                }
            )

        # Copy the images over.
        out_dir = os.path.join(self.data_processed_dir, dataset_name)
        out_image_dir = os.path.join(out_dir, "images")
        os.makedirs(out_image_dir, exist_ok=True)
        out_depth_dir = os.path.join(out_dir, "depth_images")
        os.makedirs(out_depth_dir, exist_ok=True)
        for image in tqdm(os.listdir(image_dir), desc="Moving Images", file=sys.stdout):
            shutil.copyfile(os.path.join(image_dir, image), os.path.join(out_image_dir, image))
        for depth in tqdm(os.listdir(depth_dir), desc="Moving Depth Images", file=sys.stdout):
            shutil.copyfile(os.path.join(depth_dir, depth), os.path.join(out_depth_dir, depth))

        # Save the annotation file.
        with open(os.path.join(out_dir, "annotations.json"), "w") as f:
            json.dump(out, f)

    def guava_disease_pakistan(self, dataset_name):
        # Get all of the images.
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        classes = os.listdir(dataset_dir)
        all_images = []
        for cls in classes:
            all_images.extend([os.path.join(dataset_dir, cls, i) for i in os.listdir(os.path.join(dataset_dir, cls))])

        # Resize all of the images.
        out_dir = os.path.join(self.data_processed_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
        for image in tqdm(all_images, "Resizing Images"):
            out_image = image.replace("/original/", "/processed/")
            im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            im = cv2.resize(im, (im.shape[1] // 5, im.shape[0] // 5), cv2.INTER_LINEAR)
            cv2.imwrite(out_image, im)

    def apple_detection_spain(self, dataset_name):
        # resize the dataset
        resize = 1.0

        # Read public_datasources.json to get class information
        datasource_file = os.path.join(os.path.dirname(__file__), "../_assets/public_datasources.json")
        with open(datasource_file) as f:
            data = json.load(f)
            category_info = data[dataset_name]["crop_types"]
            labels_str = []
            labels_ids = []
            for info in category_info:
                labels_str.append(category_info[info])
                labels_ids.append(int(info))

            name_converter = dict(zip(["Poma"], ["apple"]))  # src -> dst
            label2id = dict(zip(labels_str, labels_ids))

        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        ann_dir = os.path.join(dataset_dir, "preprocessed data/square_annotations1")

        # Get image file and xml file
        all_files = get_file_list(ann_dir)
        anno_files = [os.path.join(ann_dir, x) for x in all_files if "xml" in x]
        img_files = [x.replace(".xml", "hr.jpg").replace("square_annotations1", "images") for x in anno_files]

        # Process annotation files
        save_dir_anno = os.path.join(self.data_processed_dir, dataset_name, "annotations")
        create_dir(save_dir_anno)
        output_json_file = os.path.join(save_dir_anno, "instances.json")

        # Process image files
        output_img_path = os.path.join(self.data_processed_dir, dataset_name, "images")
        create_dir(output_img_path)

        general_info = {
            "description": "KFuji RGB-DS database",
            "url": "http://www.grap.udl.cat/en/publications/KFuji_RGBDS_database.html",
            "version": "1.0",
            "year": 2018,
            "contributor": "Gené-Mola J, Vilaplana V, Rosell-Polo JR, Morros JR, Ruiz-Hidalgo J, Gregorio E",
            "date_created": "2018/10/19",
        }

        convert_xmls_to_cocojson(
            general_info,
            annotation_paths=anno_files,
            img_paths=img_files,
            label2id=label2id,
            name_converter=name_converter,
            output_jsonpath=output_json_file,
            output_imgpath=output_img_path,
            extract_num_from_imgid=True,
        )

    def apple_detection_drone_brazil(self, dataset_name):
        # Get the data directory and rename it if necessary.
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            fallback = os.path.join(self.data_original_dir, "thsant-add256-68d2f88")  # noqa
            if os.path.exists(fallback):
                os.rename(fallback, dataset_dir)

        # Get all of the images which have valid annotations.
        with open(os.path.join(dataset_dir, "all.json"), "r") as f:
            original_annotations = json.load(f)
        valid_annotations = {k: v for k, v in original_annotations.items() if v != []}

        # Construct the `images` part of the COCO JSON.
        image_coco = []
        image_id_map = {}
        image_dir = os.path.join(dataset_dir, "images")
        for idx, image_name in tqdm(
            enumerate(valid_annotations.keys()),
            desc="Parsing Images",
            total=len(valid_annotations),
        ):
            height, width = cv2.imread(os.path.join(image_dir, image_name)).shape[:2]
            image_coco.append({"file_name": image_name, "height": height, "width": width, "id": idx})
            image_id_map[image_name] = idx

        # Construct the `annotations` part of the COCO JSON.
        annotation_idx = 0
        annotation_coco = []
        for image_name, annotation_list in valid_annotations.items():
            for annotation in annotation_list:
                # Coordinates are in form (center_x, center_y, radius). We convert
                # these to (top left x, top left y, width, height)
                x_c, y_c, r = annotation["cx"], annotation["cy"], annotation["r"]
                x, y = x_c - r, y_c - r
                w = h = r * 2
                annotation_coco.append(
                    {
                        "area": w * h,
                        "iscrowd": 0,
                        "bbox": [x, y, w, h],
                        "category_id": 1,
                        "ignore": 0,
                        "segmentation": 0,
                        "image_id": image_id_map[image_name],
                        "id": annotation_idx,
                    }
                )
                annotation_idx += 1

        # Set up the annotation dictionary.
        category_info = [{"supercategory": "none", "id": 1, "name": "apple"}]
        all_annotation_data = {
            "images": image_coco,
            "type": "instances",
            "annotations": annotation_coco,
            "categories": category_info,
            "info": {
                "description": "apple detection dataset with drone imagery",
                "url": "https://github.com/thsant/add256/tree/zenodo-1.0",
                "version": "1.0",
                "year": 2021,
                "contributor": "Thiago T. Santos and Luciano Gebler",
                "date_created": "2021/10/2021",
            },
        }

        # Recreate the dataset and zip it
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_img_dir = os.path.join(processed_dir, "images")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(processed_img_dir, exist_ok=True)
        for path in tqdm(valid_annotations.keys(), desc="Moving Images"):
            full_path = os.path.join(image_dir, path)
            shutil.copyfile(full_path, os.path.join(processed_img_dir, os.path.basename(path)))
        with open(os.path.join(processed_dir, "annotations.json"), "w") as f:
            json.dump(all_annotation_data, f)

    def plant_doc_classification(self, dataset_name):
        category_info = self.data_sources[dataset_name]["classes"]

        # paths to original files
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "test")

        # make output dir
        output = os.path.join(self.data_processed_dir, dataset_name)
        os.makedirs(output)

        for key in category_info:
            category = category_info[key]

            # make output dir for each crop type
            output_catg_dir = os.path.join(output, category)
            os.makedirs(output_catg_dir)

            # put train and test images of same category into same folder
            train_catg_dir = os.path.join(train_dir, category)
            test_catg_dir = os.path.join(test_dir, category)

            for img_name in get_file_list(train_catg_dir):
                img = os.path.join(train_catg_dir, img_name)
                shutil.copyfile(img, os.path.join(output_catg_dir, img_name))

            if os.path.exists(test_catg_dir):
                for img_name in get_file_list(test_catg_dir):
                    img = os.path.join(test_catg_dir, img_name)
                    shutil.copyfile(img, os.path.join(output_catg_dir, img_name))

    def wheat_head_counting(self, dataset_name):
        label2id = {"Wheat Head": 1}
        dataset_dir = os.path.join(self.data_original_dir, dataset_name)
        anno_files = [
            os.path.join(dataset_dir, "competition_train.csv"),
            os.path.join(dataset_dir, "competition_test.csv"),
            os.path.join(dataset_dir, "competition_val.csv"),
        ]

        annotations = []
        for anno_file in anno_files:
            with open(anno_file, "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    img_path = os.path.join(dataset_dir, "images", row[0])
                    anno = [img_path]
                    bboxs = row[1].split(";")
                    anno.append(len(bboxs))
                    for bbox in bboxs:
                        if bbox != "no_box":
                            bbox = bbox.split(" ")
                            bbox.append("1")
                            anno.append(bbox)
                    annotations.append(anno)

        # Define path to processed annotation files
        output_json_file = os.path.join(self.data_processed_dir, dataset_name, "annotations.json")

        # Create directory for processed image files
        output_img_path = os.path.join(self.data_processed_dir, dataset_name, "images")
        create_dir(output_img_path)

        general_info = {
            "description": "Global Wheat Head Detection (GWHD) dataset",
            "url": "http://www.global-wheat.com/",
            "version": "1.0",
            "year": 2021,
            "contributor": "David, Etienne and Madec, Simon and Sadeghi-Tehran, Pouria and Aasen, Helge and Zheng, Bangyou and Liu, Shouyang and Kirchgessner, Norbert and Ishikawa, Goro and Nagasawa, Koichi and Badhon, Minhajul A and others",
            "date_created": "2021/7/12",
        }

        convert_bbox_to_coco(
            annotations,
            label2id,
            output_json_file,
            output_img_path,
            general_info,
            resize=512 / 1024,
        )

    def peachpear_flower_segmentation(self, dataset_name):
        # Create processed directories
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        os.makedirs(processed_dir, exist_ok=True)
        processed_image_dir = os.path.join(processed_dir, "images")
        os.makedirs(processed_image_dir, exist_ok=True)
        processed_annotation_dir = os.path.join(processed_dir, "annotations")
        os.makedirs(processed_annotation_dir, exist_ok=True)

        dataset_dir = os.path.join(self.data_original_dir, dataset_name)

        # Get image files
        img_dirs = ["Peach", "Pear"]
        img_paths = []
        for img_dir in img_dirs:
            img_paths += [
                os.path.join(dataset_dir, img_dir, file_name)
                for file_name in get_file_list(os.path.join(dataset_dir, img_dir))
            ]

        # Save all images as jpg in processed directory
        for img_path in img_paths:
            processed_path = os.path.join(processed_image_dir, img_path.split("/")[-1].replace(".bmp", ".jpg"))
            img = cv2.imread(img_path)
            cv2.imwrite(processed_path, img)

        # Get annotation files
        anno_dirs = ["PeachLabels", "PearLabels"]
        anno_paths = []
        for anno_dir in anno_dirs:
            anno_paths += [
                os.path.join(dataset_dir, anno_dir, file_name)
                for file_name in get_file_list(os.path.join(dataset_dir, anno_dir))
            ]

        # Transform mask and save to processed directory
        for anno_path in anno_paths:
            img = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
            img = np.where(img[:] == 255, 1, 0)
            processed_path = os.path.join(processed_annotation_dir, anno_path.split("/")[-1])
            cv2.imwrite(processed_path, img)

    def ghai_romaine_detection(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir, dataset_name)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, "images")
        os.makedirs(processed_image_dir, exist_ok=True)

        # Move images
        for image in glob.glob(os.path.join(original_dir, "*.jpg")):
            shutil.move(image, processed_image_dir)
        shutil.move(
            os.path.join(original_dir, "coco.json"),
            os.path.join(processed_dir, "annotations.json"),
        )

    def ghai_green_cabbage_detection(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir, dataset_name)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, "images")
        os.makedirs(processed_image_dir, exist_ok=True)

        # Move images
        for image in glob.glob(os.path.join(original_dir, "*.jpg")):
            shutil.move(image, processed_image_dir)
        shutil.move(
            os.path.join(original_dir, "coco.json"),
            os.path.join(processed_dir, "annotations.json"),
        )

    def ghai_iceberg_lettuce_detection(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir, dataset_name)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, "images")
        os.makedirs(processed_image_dir, exist_ok=True)

        # Move images
        for image in glob.glob(os.path.join(original_dir, "*.jpg")):
            shutil.move(image, processed_image_dir)
        shutil.move(
            os.path.join(original_dir, "coco.json"),
            os.path.join(processed_dir, "annotations.json"),
        )

    def riseholme_strawberry_classification_2021(self, dataset_name):
        # Create processed data directory.
        original_dir = os.path.join(self.data_original_dir, "Riseholme-2021-main", "Data")
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)

        # Load all of the individual images and keep a mapping to their corresponding directory.
        images = {
            "anomalous": get_file_list(os.path.join(original_dir, "Anomalous")),
            **{
                dir_.lower(): get_file_list(os.path.join(original_dir, "Normal", dir_))
                for dir_ in os.listdir(os.path.join(original_dir, "Normal"))
            },
        }

        # Create the output file structure.
        for class_name, image_set in images.items():
            class_dir = os.path.join(processed_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for file in image_set:
                shutil.copyfile(file, os.path.join(class_dir, os.path.basename(file)))

    def ghai_broccoli_detection(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir, dataset_name)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, "images")
        os.makedirs(processed_image_dir, exist_ok=True)

        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, "*.jpg"))):
            shutil.move(image, processed_image_dir)
        shutil.move(
            os.path.join(original_dir, "coco.json"),
            os.path.join(processed_dir, "annotations.json"),
        )

    def ghai_strawberry_fruit_detection(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir, dataset_name)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, "images")
        os.makedirs(processed_image_dir, exist_ok=True)

        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, "*.jpg"))):
            shutil.move(image, processed_image_dir)
        shutil.move(
            os.path.join(original_dir, "coco.json"),
            os.path.join(processed_dir, "annotations.json"),
        )

    def vegann_multicrop_presence_segmentation(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir, dataset_name)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, "images")
        os.makedirs(processed_image_dir, exist_ok=True)
        processed_annotation_dir = os.path.join(processed_dir, "annotations")
        os.makedirs(processed_annotation_dir, exist_ok=True)

        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, "images", "*.png"))):
            shutil.copyfile(image, os.path.join(processed_image_dir, os.path.basename(image)))

        # Read annotations
        for annotation_file in tqdm(glob.glob(os.path.join(original_dir, "annotations", "*.png"))):
            annotation = cv2.imread(annotation_file, cv2.IMREAD_UNCHANGED)
            annotation = np.where(annotation == 255, 1, 0)
            cv2.imwrite(
                os.path.join(processed_annotation_dir, os.path.basename(annotation_file)),
                annotation,
            )

        # Read the CSV file containing the splits
        split_csv = pd.read_csv(os.path.join(original_dir, "VegAnn_dataset.csv"), sep=";")

        # Get the `Name` and `TVT-split{n}` columns for each n, and save the splits to a folder
        splits_folder = os.path.join(processed_dir, ".splits")
        os.makedirs(splits_folder, exist_ok=True)
        column_pairs = [["Name", f"TVT-split{i}"] for i in range(1, 5 + 1)]

        splits = {}
        for column_pair in column_pairs:
            columns = split_csv[column_pair]
            train_images = columns[columns[column_pair[1]] == "Training"]["Name"]
            test_images = columns[columns[column_pair[1]] == "Test"]["Name"]
            splits[column_pair[1]] = {
                "train": {os.path.join("images", i): os.path.join("annotations", i) for i in train_images},
                "val": {},
                "test": {os.path.join("images", i): os.path.join("annotations", i) for i in test_images},
            }

        # Save each split to a JSON file
        for split_name, split in splits.items():
            with open(os.path.join(splits_folder, f"{split_name}.json"), "w") as f:
                json.dump(split, f)

    def embrapa_wgisd_grape_detection(self, dataset_name):
        """Preprocesses your grape dataset (Chardonnay, PinotGris, PinotNoir) from YOLO format to COCO format."""
        base_path = os.path.join(self.data_original_dir)

        # Get all the dataset folders (Chardonnay, PinotGris, PinotNoir)
        dataset_folders = ['Chardonnay', 'PinotGris', 'PinotNoir']

        # Prepare the COCO annotation dictionary
        coco_annotation = {
            "images": [],
            "annotations": [],
            "categories": [{
                "id": 1,  # Category ID for the object (e.g., grape)
                "name": "grape",  # You can change this to match your dataset's object
                "supercategory": "fruit"
            }]
        }

        annotation_id = 1
        image_id = 1

        # Create the output directories for processed images and annotations
        output_base_dir = os.path.join(self.data_processed_dir, dataset_name)
        output_img_dir = os.path.join(output_base_dir, 'images')
        os.makedirs(output_img_dir, exist_ok=True)

        # Iterate through each dataset folder (Chardonnay, PinotGris, PinotNoir)
        for folder in dataset_folders:
            images_dir = os.path.join(base_path, folder, 'images')
            labels_dir = os.path.join(base_path, folder, 'labels')

            # Check if the directories exist
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print(f"Error: Images or labels directory does not exist in {images_dir} or {labels_dir}")
                continue

            # Create output directory for this folder inside the processed folder
            output_folder_img_dir = os.path.join(output_img_dir, folder)
            os.makedirs(output_folder_img_dir, exist_ok=True)

            # Process all images and their corresponding YOLO labels
            for img_file in tqdm(os.listdir(images_dir), desc=f"Processing images for {folder}"):
                if img_file.endswith(('jpg', 'jpeg', 'png')):
                    image_path = os.path.join(images_dir, img_file)
                    label_file = img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                    label_path = os.path.join(labels_dir, label_file)

                    # Read the image to get its dimensions
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error: Could not open image file {image_path}")
                        continue
                    img_height, img_width = img.shape[:2]

                    # Add image information to COCO structure
                    image_info = {
                        "file_name": f"{folder}/{img_file}",  # Include folder name in file path
                        "height": img_height,
                        "width": img_width,
                        "id": image_id
                    }
                    coco_annotation['images'].append(image_info)

                    # Read the YOLO label file and convert to COCO format
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            lines = f.readlines()

                        # Prepare annotations in COCO format
                        for line in lines:
                            elements = line.strip().split()
                            class_id = int(elements[0])  # class_id from YOLO
                            x_center, y_center, bbox_width, bbox_height = map(float, elements[1:])

                            # Convert to COCO format bounding box
                            bbox = yolo_to_coco_bbox(x_center, y_center, bbox_width, bbox_height, img_width, img_height)

                            annotation = {
                                "image_id": image_id,  # Reference to the image ID
                                "bbox": bbox,  # COCO bounding box [xmin, ymin, width, height]
                                "category_id": 1,  # Assuming a single category "grape"
                                "id": annotation_id,  # Unique annotation ID
                                "area": bbox[2] * bbox[3],  # width * height
                                "iscrowd": 0,
                                "segmentation": []
                            }
                            coco_annotation['annotations'].append(annotation)
                            annotation_id += 1
                    else:
                        print(f"Warning: No label file found for {img_file}")

                    # Copy the image to the processed directory
                    shutil.copyfile(image_path, os.path.join(output_folder_img_dir, img_file))

                    # Increment image ID
                    image_id += 1

        # Save the final COCO annotations to the correct path inside the processed folder
        output_json_file = os.path.join(output_base_dir, 'annotations.json')
        with open(output_json_file, 'w') as json_file:
            json.dump(coco_annotation, json_file, indent=4)

        print(f"COCO annotations saved to {output_json_file}")

    def plant_doc_detection(self, dataset_name):
        # Resize the dataset (if necessary)
        resize = 1.0

        # Read public_datasources.json to get class information
        datasource_file = os.path.join(os.path.dirname(__file__), "../_assets/public_datasources.json")
        with open(datasource_file) as f:
            data = json.load(f)
            category_info = data[dataset_name]['classes']  # This will give us the class information
            labels_str = []
            labels_ids = []
            for info in category_info:
                labels_str.append(category_info[info])
                labels_ids.append(int(info))

            # No name conversion in this case, unless you need to remap class names
            name_converter = None
            label2id = dict(zip(labels_str, labels_ids))  # Map class names to their respective IDs

        # Set paths to dataset and annotations
        dataset_dir = os.path.join(self.data_original_dir)
        ann_dir = dataset_dir  # Both images and XMLs are in the same directory

        # Get image file and xml file
        all_files = os.listdir(ann_dir)
        anno_files = [os.path.join(ann_dir, x) for x in all_files if x.endswith("xml")]
        img_files = [x.replace(".xml", ".jpg") for x in anno_files]  # Assuming images are in JPG format

        # # Process annotation files
        # save_dir_anno = os.path.join(self.data_processed_dir, dataset_name, 'annotations')
        # create_dir(save_dir_anno)
        # output_json_file = os.path.join(save_dir_anno, 'instances.json')

        # Process image files
        output_img_path = os.path.join(self.data_processed_dir, dataset_name, 'images')
        create_dir(output_img_path)

        # General information for the COCO JSON format
        general_info = {
            "description": "PlantDoc Object Detection Dataset",
            "url": "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "PlantDoc",
            "date_created": "2024/10/17"
        }

        # Save the COCO JSON file directly in the processed directory
        output_json_file = os.path.join(self.data_processed_dir, dataset_name, 'annotations.json')

        # Convert the XML annotations to COCO format using your existing function
        convert_xmls_to_cocojson(
            general_info=general_info,
            annotation_paths=anno_files,
            img_paths=img_files,
            label2id=label2id,
            name_converter=name_converter,
            output_jsonpath=output_json_file,
            output_imgpath=output_img_path,
            extract_num_from_imgid=False
        )

        print(f"Preprocessing completed! Annotations saved to {output_json_file}") 

    def growliflower_cauliflower_segmentation(self, dataset_name):
        """
        Preprocess the cauliflower dataset by merging mask images and moving them to the new directory.
        
        Args:
            self: Instance of the class where this function belongs.
            dataset_name (str): Name of the dataset.
        """
        # Paths to the dataset
        dataset_dir = os.path.join(self.data_original_dir)
        images_dir = os.path.join(dataset_dir, 'images')  # Folder with original images
        masks_dir = os.path.join(dataset_dir, 'annotations')  # Folder with the mask subfolders
        
        # Output directory where preprocessed data will be stored
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        processed_annotation_dir = os.path.join(processed_dir, 'annotations')
        
        # Create output directories if they don't exist
        os.makedirs(processed_image_dir, exist_ok=True)
        os.makedirs(processed_annotation_dir, exist_ok=True)
        
        # Define the mask class mappings (assign unique labels to each type)
        mask_classes = {
            'maskLeaves': 1,
            'maskPlants': 2,
            'maskStems': 3,
            'maskVoid': 4
        }

        # Get the list of image filenames
        image_files = sorted(os.listdir(images_dir))
        
        for idx, image_file in enumerate(tqdm(image_files, desc="Processing Cauliflower Dataset")):
            image_name = os.path.splitext(image_file)[0]  # Get the base image name without extension
            image_path = os.path.join(images_dir, image_file)

            # Find the corresponding masks for each class
            mask_paths = {}
            for mask_class in mask_classes.keys():
                # Build mask filenames with both naming conventions
                mask_filename_no_plants = f"{image_name}_Label_NoPlants_{mask_class}.png"
                mask_filename = f"{image_name}_Label_{mask_class}.png"
                
                # Set the correct mask path
                mask_path_no_plants = os.path.join(masks_dir, mask_class, mask_filename_no_plants)
                mask_path = os.path.join(masks_dir, mask_class, mask_filename)
                
                if os.path.exists(mask_path_no_plants):
                    mask_paths[mask_class] = mask_path_no_plants
                else:
                    mask_paths[mask_class] = mask_path

            # Load the original image to get its shape (we need the height and width)
            orig_image = cv2.imread(image_path)
            if orig_image is None:
                print(f"Warning: Unable to read image {image_path}")
                continue
            image_shape = orig_image.shape[:2]  # (height, width)

            # Initialize the merged mask as a blank image (background class 0)
            merged_mask = np.zeros(image_shape, dtype=np.uint8)

            # Visualize and debug individual masks, if needed
            individual_masks = {}

            for mask_name, class_value in mask_classes.items():
                mask = cv2.imread(mask_paths[mask_name], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Unable to read mask {mask_paths[mask_name]}")
                    continue
                individual_masks[mask_name] = mask  # Store for visualization
                
                # Debugging: Check unique values in each mask
                print(f"{mask_name} unique values for {image_name}: {np.unique(mask)}")
                
                # Binarize the mask for leaf mask based on the value 60
                if mask_name == 'maskLeaves':
                    binarized_mask = np.where(mask == 60, 1, 0).astype(np.uint8)
                else:
                    # For other masks, threshold based on the condition > 60
                    binarized_mask = np.where(mask > 60, 1, 0).astype(np.uint8)
                # Merge logic: void mask only applies where there's no other mask
                if class_value == 4:  # Void class
                    merged_mask[(merged_mask == 0) & (binarized_mask > 0)] = class_value
                else:
                    merged_mask[(binarized_mask > 0)] = class_value

            # Debugging Step: Check unique values in the merged mask
            unique_values = np.unique(merged_mask)
            print(f"Image {idx+1}/{len(image_files)}: {image_file} - Unique mask values: {unique_values}")

            # Save the original image and the merged mask to the new directory
            out_image_path = os.path.join(processed_image_dir, image_file)
            out_mask_path = os.path.join(processed_annotation_dir, f"{image_name}.png")

            cv2.imwrite(out_image_path, orig_image)  
            cv2.imwrite(out_mask_path, merged_mask)  

    def strawberry_detection_2023(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        os.makedirs(processed_image_dir, exist_ok = True)
        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, '*.jpg'))):
            shutil.move(image, processed_image_dir)
        shutil.move(os.path.join(original_dir, 'coco.json'),
                    os.path.join(processed_dir, 'annotations.json'))
        
    def strawberry_detection_2022(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        os.makedirs(processed_image_dir, exist_ok = True)
        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, '*.jpg'))):
            shutil.move(image, processed_image_dir)
        shutil.move(os.path.join(original_dir, 'coco.json'),
                    os.path.join(processed_dir, 'annotations.json'))
        
    def almond_harvest_2021(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        os.makedirs(processed_image_dir, exist_ok = True)
        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, '*.jpg'))):
            shutil.move(image, processed_image_dir)
        shutil.move(os.path.join(original_dir, 'coco.json'),
                    os.path.join(processed_dir, 'annotations.json'))

    def almond_bloom_2023(self, dataset_name):
        # Create processed directories
        original_dir = os.path.join(self.data_original_dir)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        os.makedirs(processed_image_dir, exist_ok = True)
        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, '*.jpg'))):
            shutil.move(image, processed_image_dir)
        shutil.move(os.path.join(original_dir, 'coco.json'),
                    os.path.join(processed_dir, 'annotations.json'))
        
    def gemini_flower_detection_2022(self, dataset_name):
        original_dir = os.path.join(self.data_original_dir)
        print(original_dir)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        os.makedirs(processed_image_dir, exist_ok = True)
        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, 'images', '*.jpg'))):
            shutil.move(image, processed_image_dir)
        shutil.move(os.path.join(original_dir, 'coco.json'),
                    os.path.join(processed_dir, 'annotations.json'))
        
    def gemini_leaf_detection_2022(self, dataset_name):
        original_dir = os.path.join(self.data_original_dir)
        print(original_dir)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        os.makedirs(processed_image_dir, exist_ok = True)
        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, 'images', '*.jpg'))):
            shutil.move(image, processed_image_dir)
        shutil.move(os.path.join(original_dir, 'coco.json'),
                    os.path.join(processed_dir, 'annotations.json'))
    
    def gemini_pod_detection_2022(self, dataset_name):
        original_dir = os.path.join(self.data_original_dir)
        print(original_dir)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        os.makedirs(processed_image_dir, exist_ok = True)
        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, 'images', '*.jpg'))):
            shutil.move(image, processed_image_dir)
        shutil.move(os.path.join(original_dir, 'coco.json'),
                    os.path.join(processed_dir, 'annotations.json'))
        
    def gemini_plant_detection_2022(self, dataset_name):
        original_dir = os.path.join(self.data_original_dir)
        print(original_dir)
        processed_dir = os.path.join(self.data_processed_dir, dataset_name)
        processed_image_dir = os.path.join(processed_dir, 'images')
        os.makedirs(processed_image_dir, exist_ok = True)
        # Move images
        for image in tqdm(glob.glob(os.path.join(original_dir, 'images', '*.jpg'))):
            shutil.move(image, processed_image_dir)
        shutil.move(os.path.join(original_dir, 'coco.json'),
                    os.path.join(processed_dir, 'annotations.json'))

    def paddy_disease_classification(self, dataset_name):
        pass
    
    def onion_leaf_classification(self, dataset_name):
        pass

    def chilli_leaf_classification(self, dataset_name):
        pass

    def orange_leaf_disease_classification(self, dataset_name):
        pass

    def papaya_leaf_disease_classification(self, dataset_name):
        pass

    def blackgram_plant_leaf_disease_classification(self, dataset_name):
        pass

    def arabica_coffee_leaf_disease_classification(self, dataset_name):
        pass

    def banana_leaf_disease_classification(self, dataset_name):
        pass

    def coconut_tree_disease_classification(self, dataset_name):
        pass

    def rice_leaf_disease_classification(self, dataset_name):
        pass

    def tea_leaf_disease_classification(self, dataset_name):
        pass

    def betel_leaf_disease_classification(self, dataset_name):
        pass

    def java_plum_leaf_disease_classification(self, dataset_name):
        pass

    def sunflower_disease_classification(self, dataset_name):
        pass

    def cucumber_disease_classification(self, dataset_name):
        pass

if __name__ == '__main__':
    # Initialize program arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default="../../data_new",
        help="The directory containing two sub-directories, " "`original` and `processed`, with the data.",
    )
    ap.add_argument("--dataset", type=str, help="The dataset to process.")
    args = ap.parse_args()

    # Execute the preprocessing.
    p = PublicDataPreprocessor(os.path.abspath(args.data_dir))
    print("Processing dataset")
    p.preprocess(args.dataset)
    print("Converting dataset")
    # os.chdir(f'{args.data_dir}/processed')
    # os.system(f'zip -r {args.dataset}.zip {args.dataset} -x ".*" -x "__MACOSX"')
