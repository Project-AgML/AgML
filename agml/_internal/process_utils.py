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
Preprocess utilities for the code in `preprocess.py`.
"""

import os
import sys
import csv
import json
import shutil
from typing import Dict, List
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from PIL import Image
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

from agml.utils.logging import tqdm


def read_txt_file(file_name, delimiter = ' ', header = False):
    with open(file_name, newline = '\n') as txt_file:
        txt_reader = csv.reader(txt_file, delimiter = delimiter)
        txt_lines = []
        headline = None
        for idx, line in enumerate(txt_reader):
            if header and idx == 0:
                headline = line
                continue
            line = [x.strip() for x in line if x.strip()]  # To remove blank elements
            txt_lines.append(line)

        return txt_lines, headline

def get_label2id(labels_str: str) -> Dict[str, int]:
    """id is 1 start"""
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If wanting to use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If wanting to use annotation ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]
    return ann_paths

def get_image_info_from_annoline(annotation_root, idx, resize = 1.0, add_foldername = False):
    filename = annotation_root[0].split('/')[-1]
    try:
        img = cv2.imread(annotation_root[0])

        if resize != 1.0:
            dsize = (int(img.shape[1] * resize), int(img.shape[0] * resize))
            img = cv2.resize(img, dsize)

        size = img.shape
        width = size[1]
        height = size[0]

        if add_foldername:
            filename = "{folder}_{img_name}".format(folder = annotation_root[0].split('/')[-2],
                                                    img_name = annotation_root[0].split('/')[-1])

        image_info = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': idx  # Use image order
        }

    except Exception as e:
        print(e)
        print("Cannot open {file}".format(file = annotation_root[0]))
        image_info = None
        img = None

    return image_info, img

def get_image_info(annotation_root, idx, resize = 1.0, add_foldername = False):
    path = annotation_root.findtext('path')
    if path is None or True:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)

    try:
        img = cv2.imread(filename)

        if resize != 1.0:
            dsize = [int(img.shape[1] * resize), int(img.shape[0] * resize)]
            img = cv2.resize(img, dsize)

        size = img.shape
        width = size[1]
        height = size[0]

        if add_foldername:
            filename = "{folder}_{img_name}".format(folder = filename.split('/')[-2],
                                                    img_name = filename.split('/')[-1])

        image_info = {
            'file_name': filename.split('/')[-1],
            'height': height,
            'width': width,
            'id': idx  # Use image order
        }

    except Exception as e:
        print(e)
        print("Cannot open {file}".format(file = annotation_root[0]))
        image_info = None
        img = None

    return image_info, img

# Sourced from https://github.com/roboflow-ai/voc2coco.git
def get_coco_annotation_from_annoline(obj, resize = 1.0):
    # Try the sub-label first
    category_id = int(obj[4])
    xmin = int(float(obj[0]) * resize)
    ymin = int(float(obj[1]) * resize)
    xmax = int(float(obj[2]) * resize)
    ymax = int(float(obj[3]) * resize)
    assert xmax > xmin and ymax > ymin, \
        f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin + 1
    o_height = ymax - ymin + 1
    return {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }

def get_coco_annotation_from_obj(obj, label2id, name_converter = None):
    # Try to sub-label first
    label = obj.findtext('subname')
    if label is None:
        label = obj.findtext('name')
        if name_converter:
            if label in name_converter:
                label = name_converter[label]
        if label not in label2id:
            return []
    category_id = label2id[label]

    bndbox = obj.find('bndbox')
    if bndbox is None:
        bndbox = obj.find('bbox')
        if bndbox is None:
            return None

    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, \
        f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    return {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }

def convert_bbox_to_coco(annotation: List[str],
                         label2id: Dict[str, int],
                         output_jsonpath: str,
                         output_imgpath: str,
                         general_info,
                         image_id_list = None,
                         bnd_id_list = None,
                         get_label_from_folder = False,
                         resize = 1.0,
                         add_foldername = False,
                         extract_num_from_imgid = False):
    """
    Annotation Format
    -----------------
    "image name" "the number of bounding boxes(bb)" "x1" "y1" "x2" "y2" "label" "score" "x1" "y1" "x2" "y2" ...
    
    For example, the following line:
    
    TRAIN_RGB/n12710693_12225.png 5 515 68 759 285 2 1.000 624 347 868 582 2 1.000 480 488 693 712 2 1.000 44 433 268 657 2 1.000 112 198 342 401 2 1.000
    
    will become: 
    
    image name=TRAIN_RGB/n12710693_12225.png
    the number of bb=5
    x1=515
    y1=68
    x2=759
    y2=285
    label=2 "0=background, 1=capsicum, 2=rockmelon..."
    score=1.000
    
    Reference: https://drive.google.com/drive/folders/1CmsZb1caggLRN7ANfika8WuPiywo4mBb
    """
    output_json_dict = {
        "images": [], "type": "instances", "annotations": [],
        "categories": [], 'info': general_info}

    # TODO: Use multi thread to boost up the speed
    print("Converting annotations into COCO JSON and process the images")
    for img_idx, anno_line in enumerate(tqdm(annotation)):

        if image_id_list:
            img_unique_id = image_id_list[img_idx]
        else:
            if extract_num_from_imgid:
                filename = anno_line[0].split('/')[-1]
                img_unique_id = int(''.join(filter(str.isdigit, filename)))
            else:
                img_unique_id = img_idx + 1

        img_info, img = get_image_info_from_annoline(
            annotation_root = anno_line, idx = img_unique_id,
            resize = resize, add_foldername = add_foldername)

        if img_info:
            output_json_dict['images'].append(img_info)

            bbox_cnt = int(anno_line[1])
            if bbox_cnt > 0:
                ann_reshape = np.reshape(anno_line[2:], (bbox_cnt, -1))
                for bnd_idx, obj in enumerate(ann_reshape):
                    if get_label_from_folder:
                        # Change label based on folder
                        try:
                            category_name = anno_line[0].split('/')[-3]
                            if category_name not in label2id:
                                raise
                        except:
                            try:
                                category_name = anno_line[0].split('/')[-2]
                                if category_name not in label2id:
                                    raise
                            except Exception as e:
                                raise e

                        if len(obj) < 5:
                            obj = np.append(obj, label2id[category_name])
                        else:
                            obj[4] = label2id[category_name]
                    else:
                        pass

                    try:
                        ann = get_coco_annotation_from_annoline(
                            obj = obj, resize = resize)
                    except:
                        ann = None

                    if ann:
                        if bnd_id_list:
                            bnd_idx = bnd_id_list[img_idx][bnd_idx]
                        else:
                            bnd_idx + 1 # noqa
                        ann.update({'image_id': img_info['id'], 'id': bnd_idx})
                        output_json_dict['annotations'].append(ann)

            img_name = img_info['file_name']
            dest_path = os.path.join(output_imgpath, img_name)
            try:
                if resize == 1.0:
                    shutil.copyfile(anno_line[0], dest_path)
                else:
                    cv2.imwrite(dest_path, img)
            except: # Cannot copy the image file
                pass

        else: # Not valid image => Delete from annotation
            pass

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

    return output_json_dict

def convert_xmls_to_cocojson(general_info,
                             annotation_paths: List[str],
                             img_paths: List[str],
                             label2id: Dict[str, int],
                             name_converter,
                             output_jsonpath: str,
                             output_imgpath: str,
                             extract_num_from_imgid: bool = True):
    """Converts the XML annotation format into COCO JSON."""
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
        "info": general_info
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for img_idx, a_path in enumerate(tqdm(annotation_paths)):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        if extract_num_from_imgid:
            filename = a_path.split('/')[-1]
            img_unique_id = int(''.join(filter(str.isdigit, filename)))
        else:
            img_unique_id = img_idx + 1

        if len(img_paths) == len(annotation_paths):
            ann_root.find("filename").text = img_paths[img_idx]

        img_info, img = get_image_info(
            annotation_root = ann_root, idx = img_unique_id,
            resize = 1.0, add_foldername = False)
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(
                obj = obj, label2id = label2id, name_converter = name_converter)
            if ann:
                ann.update({'image_id': img_info['id'], 'id': bnd_id})
                output_json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

        # Process images
        img_name = img_info['file_name']
        dest_path = os.path.join(output_imgpath, img_name)
        try:
            cv2.imwrite(dest_path, img)
        except: # Cannot copy the image file
            pass

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

# Reference: https://www.immersivelimit.com/create-coco-annotations-from-scratch
def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

    return sub_masks

# Reference: https://www.immersivelimit.com/create-coco-annotations-from-scratch
def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation = 'low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology = False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def create_sub_mask_annotation_per_bbox(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    sub_mask_np = np.array(sub_mask)
    contours = measure.find_contours(sub_mask_np, 0.5, positive_orientation = 'low')

    segmentations = []
    polygons = []
    annotations = []
    for idx, contour in enumerate(contours):
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology = False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

        if poly.area > 0:
            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon([poly])
            x, y, max_x, max_y = multi_poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = multi_poly.area

            annotations.append({
                'segmentation': [segmentation],
                'iscrowd': is_crowd,
                'image_id': image_id,
                'category_id': category_id,
                'id': annotation_id + idx,
                'bbox': bbox,
                'area': area
            })

    return annotations


def mask_annotation_per_bbox(anno_line, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    segmentations = []
    polygons = []
    annotations = []
    mask_data = json.loads(anno_line[5])

    if len(mask_data['all_points_x']) > 3:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        contour = []
        for i in range(len(mask_data['all_points_x'])):
            contour.append([int(mask_data['all_points_x'][i]), int(mask_data['all_points_y'][i])])

        contour = np.array(contour)
        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology = False)
        polygons.append(poly)
        # Only need the contours, not the polygon itself:
        # segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentation = contour.ravel().tolist()

        segmentations.append(segmentation)

        if poly.area > 0:
            # Combine the polygons to calculate the bounding box and area
            x, y, max_x, max_y = poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = poly.area

            annotation = {
                'segmentation': [segmentation],
                'iscrowd': is_crowd,
                'image_id': image_id,
                'category_id': category_id,
                'id': annotation_id,
                'bbox': bbox,
                'area': area
            }
            annotations.append(annotation)

    return annotations

def move_segmentation_dataset(
        out_dir, dataset_name, train_images, annotation_images, train_dir,
        annotation_dir, train_preprocess_fn = None, annotation_preprocess_fn = None):
    """Moves segmentation images and annotations to a new location."""
    processed_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(processed_dir, exist_ok = True)
    processed_image_dir = os.path.join(processed_dir, 'images')
    os.makedirs(processed_image_dir, exist_ok = True)
    processed_annotation_dir = os.path.join(processed_dir, 'annotations')
    os.makedirs(processed_annotation_dir, exist_ok = True)
    for image_path, annotation_path in zip(
            tqdm(train_images, desc = "Processing Images",
                 file = sys.stdout), annotation_images):
        orig_image_path = os.path.join(train_dir, image_path)
        orig_annotation_path = os.path.join(annotation_dir, annotation_path)
        out_image_path = os.path.join(
            processed_image_dir, image_path)
        out_label_path = os.path.join(
            processed_annotation_dir, annotation_path)
        if train_preprocess_fn is None:
            shutil.copyfile(orig_image_path, out_image_path)
        else:
            train_preprocess_fn(orig_image_path, out_image_path)
        if annotation_preprocess_fn is None:
            shutil.copyfile(orig_annotation_path, out_label_path)
        else:
            annotation_preprocess_fn(orig_annotation_path, out_label_path)
