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

import numpy as np

from agml.backend.tftorch import is_array_like
from agml.utils.general import resolve_list_value


def _resolve_coco_annotations(annotations):
    """Resolves COCO annotations to a standard format.

    Generally, COCO annotations will either be represented as a
    dictionary which contains multiple elements for each of its keys,
    e.g. multiple bounding boxes and areas, or a list of multiple COCO
    dictionaries. This method resolves it into the former case.
    """
    if is_array_like(annotations) and not isinstance(annotations, np.ndarray):
        return annotations.numpy()
    if isinstance(annotations, np.ndarray):
        return annotations
    if isinstance(annotations, list):
        if not isinstance(annotations[0], dict):
            return annotations
        if len(annotations) == 1:
            return annotations
        annotation = {'bboxes': [], 'labels': [], 'area': [],
                      'image_id': "", 'iscrowd': [], 'segmentation': []}
        for a_set in annotations:
            annotation['bboxes'].append(a_set['bbox'])
            annotation['labels'].append(a_set['category_id'])
            annotation['iscrowd'].append(a_set['iscrowd'])
            annotation['segmentation'].append(a_set['segmentation'])
            annotation['area'].append(a_set['area'])
            annotation['image_id'] = a_set['image_id']
        for key, value in annotation.items():
            out = np.array(value)
            if np.isscalar(out):
                out = out.item()
            annotation[key] = out
        return annotation
    elif isinstance(annotations, dict):
        return annotations
    else:
        raise TypeError(
            "Expected either a single COCO annotation "
            "dictionary or a list of multiple dictionaries.")


def coco_to_bboxes(annotations):
    """Extracts the bounding boxes and labels from COCO JSON annotations.

    Given either a list of COCO JSON annotations, or a single
    dictionary with multiple values, this method will extract the
    bounding boxes and category labels from the annotations and
    return just the bounding boxes and category labels in two arrays.

    Parameters
    ----------
    annotations : {list, dict}
        The COCO JSON annotations in list/dict format.

    Returns
    -------
    Two arrays consisting of the bounding boxes and labels.
    """
    annotations = _resolve_coco_annotations(annotations)
    return annotations['bboxes'], annotations['labels']


def convert_bbox_format(annotations_or_bboxes, fmt):
    """Converts bounding box formats for COCO JSON and others.

    This method converts the format of bounding boxes as specified
    in the 'fmt' argument, which describes the format of the bounding
    boxes passed to the 'annotations_or_bboxes' argument. From there,
    it will convert to the standard COCO JSON bounding box format,
    namely ('x1', 'y1' 'width', 'height'). This method supports the
    following conversions (note that 'x1' and 'y1' are the top-left
    coordinates, and 'x2' and 'y2' are top-right):

    1. ('x1', 'x2', 'y1', 'y2') to COCO JSON.
    2. ('x_min', 'y_min', 'x_max', 'y_max') to COCO JSON. This format
        can also be passed with the simple string `pascal_voc`.
    3. ('x_min', 'y_min', 'width', 'height') to COCO JSON.

    Note that the variables in the bounding boxes in the above format
    can be in any order, this just needs to be reflected in the 'fmt'
    argument, and it should contain some combination of the above.

    Parameters
    ----------
    annotations_or_bboxes : {np.ndarray, list, dict}
        Either a COCO JSON annotation dictionary, or a numpy array/list
        with all of the annotations.
    fmt : {list, tuple}
        A list or tuple with one of the above formats.

    Returns
    -------
    The initial argument type (either dict or array) with the bounding
    boxes formatted in the COCO JSON format.
    """
    annotations_or_bboxes = _resolve_coco_annotations(annotations_or_bboxes)
    if isinstance(annotations_or_bboxes, dict):
        annotations = annotations_or_bboxes['bboxes']
    else:
        annotations = annotations_or_bboxes
        if isinstance(annotations[0], (int, float)):
            annotations = [annotations]
    if isinstance(fmt, str):
        if 'voc' in fmt or 'pascal' in fmt:
            fmt = 'x_min y_min x_max y_max'
        elif 'efficientdet' in fmt or 'effdet' in fmt:
            fmt = 'y_min x_min y_max x_max'
        if ',' in fmt:
            fmt = fmt.split(',')
        else:
            fmt = fmt.split(' ')
    if len(fmt) != 4:
        raise ValueError(f"Argument 'fmt' should contain 4 values, got {len(fmt)}.")
    
    # Define all of the intermediate conversion methods
    def _x1_x2_y1_y2_to_coco(annotation): # noqa
        x1, x2, y1, y2 = annotation
        width, height = abs(x2 - x1), abs(y2 - y1)
        return [x1, y1, width, height]
    def _xmin_ymin_xmax_ymax_to_coco(annotation): # noqa
        xmin, ymin, xmax, ymax = annotation
        width, height = abs(xmax - xmin), abs(ymax - ymin)
        return [xmin, ymin, width, height]
    def _xmin_ymin_width_height_to_coco(annotation): # noqa
        xmin, ymin, width, height = annotation
        x1, y1 = xmin, ymin - height
        return [x1, y1, width, height]
    def _x1_y1_width_height_to_coco(annotation): # noqa
        return annotation # This is just here for reordering.

    # Resolve the format
    fmt_bases = [['x1', 'x2', 'y1', 'y2'],
                 ['x_min', 'y_min', 'x_max', 'y_max'],
                 ['x_min', 'y_min', 'width', 'height'],
                 ['x1', 'y1', 'width', 'height']]
    fmt_map = {0: _x1_x2_y1_y2_to_coco,
               1: _xmin_ymin_xmax_ymax_to_coco,
               2: _xmin_ymin_width_height_to_coco,
               3: _x1_y1_width_height_to_coco}
    fmt_found = False
    map_fmt, select_order = None, None
    for indx, base in enumerate(fmt_bases):
        if all(i in base for i in fmt):
            fmt_found, map_fmt = True, fmt_map[indx]
            select_order = [base.index(i) for i in fmt]
    if not fmt_found:
        raise ValueError(
            f"Invalid format {fmt}, see `convert_bbox_format` "
            f"for information about valid formats.")

    # Convert the formats
    formatted_annotations = []
    for bbox in annotations:
        sorted_bbox = np.array(bbox)[select_order]
        formatted_annotations.append(map_fmt(sorted_bbox))
    formatted_annotations = np.array(formatted_annotations)
    if isinstance(annotations_or_bboxes, dict):
        res = annotations_or_bboxes.copy()
        res['bboxes'] = formatted_annotations
        return res
    return resolve_list_value(formatted_annotations)

