import os
from os import listdir
from os.path import isfile, join, isdir
import csv

from typing import Dict, List
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET
import json

import numpy as np


def get_filelist(filepath):
    return [f for f in listdir(filepath) if isfile(join(filepath, f))]


def get_dirlist(filepath):
    return [f for f in listdir(filepath) if isdir(join(filepath, f))]


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_txt_file(file_name):
    with open(file_name, newline = '\n') as txt_file:
        txt_reader = csv.reader(txt_file, delimiter = ' ')
        txt_lines = []
        for line in txt_reader:
            line = [x for x in line if x]  # To remove blank elements
            txt_lines.append(line)

        return txt_lines


def get_label2id(labels_str: str) -> Dict[str, int]:
    """id is 1 start"""

    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_image_info(annotation_root, idx):
    filename = annotation_root[0].split('/')[-1]
    try:
        img = cv2.imread(annotation_root[0])

        size = img.shape
        width = size[1]
        height = size[0]

        image_info = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': idx + 1  # Use image order
        }

    except:
        print("Cannot open {file}".format(file = annotation_root[0]))
        image_info = None

    return image_info


'''
Reference : https://github.com/roboflow-ai/voc2coco.git
'''


def get_coco_annotation_from_obj(obj, label2id):
    # Try to sublabel fist
    category_id = int(obj[4])
    xmin = int(float(obj[0])) - 1
    ymin = int(float(obj[1])) - 1
    xmax = int(float(obj[2]))
    ymax = int(float(obj[3]))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_txt_to_cocojson(annotation: List[str],
                            label2id: Dict[str, int],
                            output_jsonpath: str,
                            general_info):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
        "info": [],
    }

    output_json_dict['info'] = general_info

    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    img_id_cnt = 1

    for idx, anno_line in tqdm(enumerate(annotation)):
        img_info = get_image_info(annotation_root = anno_line, idx = idx)

        if img_info:
            # img_id = img_info['id']
            img_info['id'] = img_id_cnt
            output_json_dict['images'].append(img_info)

            bbox_cnt = int(anno_line[1])
            ann_reshape = np.reshape(anno_line[2:], (bbox_cnt, -1))
            for obj in ann_reshape:
                # Change label based on folder
                fruit_name = anno_line[0].split('/')[-3]
                if len(obj) < 5:
                    obj = np.append(obj, label2id[fruit_name])
                else:
                    obj[4] = label2id[fruit_name]

                ann = get_coco_annotation_from_obj(obj = obj, label2id = label2id)
                if ann:
                    # ann.update({'image_id': img_id, 'id': bnd_id})
                    ann.update({'image_id': img_id_cnt, 'id': bnd_id})
                    output_json_dict['annotations'].append(ann)
                    bnd_id = bnd_id + 1
            img_id_cnt = img_id_cnt + 1
        else:
            # Not valid image => Delete from anno
            # annotation.remove(anno_line)
            pass

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)