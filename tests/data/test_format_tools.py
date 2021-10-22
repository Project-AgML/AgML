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

import pytest

import agml.data as agdata
import numpy as np

@pytest.fixture
def coco_example():
    return [
        {
            "area": 5472, "iscrowd": 0,
            "bbox": [162, 648, 76, 72],
            "category_id": 1, "ignore": 0,
            "segmentation": [], "id": 4,
            "image_id": 800
        },
        {
            "area": 10712, "iscrowd": 0,
            "bbox": [1052, 547, 103, 104],
            "category_id": 1, "ignore": 0,
            "segmentation": [], "id": 5,
            "image_id": 800
        },
        {
            "area": 13924, "iscrowd": 0,
            "bbox": [1125, 560, 118, 118],
            "category_id": 1, "ignore": 0,
            "segmentation": [], "id": 6,
            "image_id": 800
        },
        {
            "area": 10200, "iscrowd": 0,
            "bbox": [456, 581, 100, 102],
            "category_id": 1, "ignore": 0,
            "segmentation": [], "id": 1,
            "image_id": 801
        },
        {
            "area": 10584, "iscrowd": 0,
            "bbox": [537, 598, 98, 108],
            "category_id": 1, "ignore": 0,
            "segmentation": [], "id": 2,
            "image_id": 801
        }
    ]

def test_coco_bbox_extraction(coco_example):
    bboxes = [b['bbox'] for b in coco_example]
    assert agdata.coco_to_bboxes(coco_example)[0].tolist() == bboxes

def test_coco_formatting_rearrange(coco_example):
    bboxes = agdata.coco_to_bboxes(coco_example)[0]
    og_bboxes = bboxes.copy()
    bboxes[:, [0, 2]] = bboxes[:, [2, 0]]
    bboxes[:, [1, 3]] = bboxes[:, [3, 1]]
    assert np.all(agdata.convert_bbox_format(
        bboxes, 'width height x1 y1') == og_bboxes)

def test_coco_formatting_minmax():
    eg_box = [[100, 500, 200, 700]]
    eg_output = [[100, 200, 400, 500]]
    assert np.all(agdata.convert_bbox_format(
        eg_box, 'x1 x2 y1 y2') == eg_output)


