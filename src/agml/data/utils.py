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
from shutil import copyfile, copytree

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


def get_image_info(annotation_root, idx, resize = 1.0):
    filename = annotation_root[0].split('/')[-1]
    try:
        img = cv2.imread(annotation_root[0])

        if resize != 1.0:
            dsize = [int(img.shape[1] * resize), int(img.shape[0] * resize)]
            img = cv2.resize(img, dsize)

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
        img = None

    return image_info, img


'''
Reference : https://github.com/roboflow-ai/voc2coco.git
'''


def get_coco_annotation_from_obj(obj, label2id, resize = 1.0):
    # Try to sublabel fist
    category_id = int(obj[4])
    xmin = int(float(obj[0]) * resize) 
    ymin = int(float(obj[1]) * resize) 
    xmax = int(float(obj[2]) * resize) 
    ymax = int(float(obj[3]) * resize) 
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin + 1
    o_height = ymax - ymin + 1
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_bbox_to_coco(annotation: List[str],
                            label2id: Dict[str, int],
                            output_jsonpath: str,
                            output_imgpath: str,
                            general_info,
                            image_id_list = None,
                            bnd_id_list = None,
                            get_label_from_folder=False,
                            resize = 1.0):
    output_json_dict = {
        "images": [], "type": "instances", "annotations": [],
        "categories": [], 'info': general_info}

    if image_id_list:
        img_id_cnt = image_id_list[0]
    else:
        img_id_cnt = 1

    for img_idx, anno_line in tqdm(enumerate(annotation)):
        img_info, img = get_image_info(annotation_root = anno_line, idx = img_id_cnt, resize=resize)

        if img_info:
            if image_id_list:
                img_info['id'] = image_id_list[img_idx]
            else:
                img_info['id'] = img_id_cnt

            output_json_dict['images'].append(img_info)

            bbox_cnt = int(anno_line[1])
            if bbox_cnt > 0:
                ann_reshape = np.reshape(anno_line[2:], (bbox_cnt, -1))
                for bnd_idx, obj in enumerate(ann_reshape):
                    if get_label_from_folder:
                        # Change label based on folder
                        try:
                            category_name = anno_line[0].split('/')[-3]
                            if category_name in label2id:
                                pass
                            else:
                                raise
                        except:
                            try:
                                category_name = anno_line[0].split('/')[-2]
                                if category_name in label2id:
                                    pass
                                else:
                                    raise
                            except:
                                raise

                        if len(obj) < 5:
                            obj = np.append(obj, label2id[category_name])
                        else:
                            obj[4] = label2id[category_name]
                    else:
                        pass

                    ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id, resize=resize)
                    if ann:
                        if bnd_id_list:
                            bnd_idx = bnd_id_list[img_idx][bnd_idx]
                        else:
                            bnd_idx + 1
                        ann.update({'image_id': img_id_cnt, 'id': bnd_idx})
                        output_json_dict['annotations'].append(ann)

            if image_id_list == None:
                img_id_cnt = img_id_cnt + 1


                           
            img_name = anno_line[0].split('/')[-1]
            dest_path = os.path.join(output_imgpath, img_name)
            try:
                if resize == 1.0:
                    copyfile(anno_line[0], dest_path)
                else:
                    cv2.imwrite(dest_path,img)
            except:
                # Cannot copy the image file
                pass
                
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