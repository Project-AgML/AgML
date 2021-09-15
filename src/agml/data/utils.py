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

def get_dirlist_nested(filepath):
    result = []
    for f in listdir(filepath):
        if isdir(join(filepath, f)):
            result.append(f)
            for ff in get_dirlist_nested(join(filepath, f)):
                result.append(join(f,ff))
    
    return result


def create_dir(dir_):
    os.makedirs(dir_, exist_ok = True)


def read_txt_file(file_name, delimiter=' '):
    with open(file_name, newline = '\n') as txt_file:
        txt_reader = csv.reader(txt_file, delimiter = delimiter)
        txt_lines = []
        for line in txt_reader:
            line = [x.strip() for x in line if x.strip()]  # To remove blank elements
            txt_lines.append(line)
        return txt_lines


def get_label2id(labels_str: list) -> Dict[str, int]:
    """Enumerates a set of string labels (starting with 1)."""
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_image_info(annotation_root, idx, resize = 1.0, make_unique_name = False):
    """Get information about an image for annotations."""
    # Check if just the image filepath is passed, or a list of annotations
    try:
        if os.path.exists(annotation_root):
            img = cv2.imread(annotation_root)
            filename = annotation_root

        else:
            filename = os.path.basename(annotation_root[0])
            img = cv2.imread(annotation_root[0])
    except:
        filename = os.path.basename(annotation_root[0])
        img = cv2.imread(annotation_root[0])

    if resize != 1.0:
        dsize = [int(img.shape[1] * resize), int(img.shape[0] * resize)]
        img = cv2.resize(img, dsize)

    try:
        size = img.shape
        width = size[1]
        height = size[0]

        if make_unique_name:
            filename = "{folder}_{img_name}".format(
                folder=annotation_root[0].split('/')[-2],
                img_name=annotation_root[0].split('/')[-1])
 
        image_info = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': idx + 1  # Use image order
        }

    except Exception as e:
        print(e)
        print("Cannot open {file}".format(file = annotation_root[0]))
        image_info = None
        img = None

    return image_info, img


'''
Reference : https://github.com/roboflow-ai/voc2coco.git
'''
def get_coco_annotation_from_obj(obj, id_name = None):
    # Calculate the area of the box
    category_id = int(obj[4])
    xmin = int(float(obj[0])) - 1
    ymin = int(float(obj[1])) - 1
    xmax = int(float(obj[2]))
    ymax = int(float(obj[3]))
    assert xmax > xmin and ymax > ymin, \
        f"Box size error!: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin

    # Construct the annotation
    ann = { # noqa
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }

    # If there are multiple annotations per image, add the ID
    if id_name is not None:
        ann['id'] = id_name
    return ann


'''
Annotaion Format
"image name" "the number of bounding boxes(bb)" "x1" "y1" "x2" "y2" "label" "score" "x1" "y1" "x2" "y2" ...
For example, the following line can be matched as

TRAIN_RGB/n12710693_12225.png 5 515 68 759 285 2 1.000 624 347 868 582 2 1.000 480 488 693 712 2 1.000 44 433 268 657 2 1.000 112 198 342 401 2 1.000

image name=TRAIN_RGB/n12710693_12225.png
the number of bb=5
x1=515
y1=68
x2=759
y2=285
label=2 "0=background, 1=capsicum, 2=rockmelon..."
score=1.000

reference  "https://drive.google.com/drive/folders/1CmsZb1caggLRN7ANfika8WuPiywo4mBb"
'''
def convert_bbox_to_coco(annotation: List[str],
                            label2id: Dict[str, int],
                            output_jsonpath: str,
                            output_imgpath: str,
                            general_info,
                            image_id_list = None,
                            bnd_id_list = None,
                            get_label_from_folder=False,
                            resize = 1.0,
                            make_unique_name=False):
    output_json_dict = {
        "images": [], "type": "instances", "annotations": [],
        "categories": [], 'info': general_info}

    if image_id_list:
        img_id_cnt = image_id_list[0]
    else:
        img_id_cnt = 1

    # TODO: Use multi thread to boost up the speed
    for img_idx, anno_line in enumerate(tqdm(annotation)):
        img_info, img = get_image_info(annotation_root=anno_line, idx=img_id_cnt, resize=resize, make_unique_name=make_unique_name)

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
                    
                    try:
                        ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id, resize=resize)
                    except:
                        ann = None

                    if ann:
                        if bnd_id_list:
                            bnd_idx = bnd_id_list[img_idx][bnd_idx]
                        else:
                            bnd_idx + 1
                        ann.update({'image_id': img_id_cnt, 'id': bnd_idx})
                        output_json_dict['annotations'].append(ann)

            if image_id_list == None:
                img_id_cnt = img_id_cnt + 1

            img_name = img_info['file_name']
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

