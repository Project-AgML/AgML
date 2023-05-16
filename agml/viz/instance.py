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

import cv2
import numpy as np

from agml.data.tools import convert_bbox_format
from agml.utils.general import resolve_tuple_values, weak_squeeze, as_scalar
from agml.viz.boxes import _resolve_proportional_bboxes
from agml.viz.tools import get_colormap, format_image


def annotate_instance_segmentation(image,
                                   segmentation = None,
                                   bboxes = None,
                                   labels = None,
                                   inplace = True,
                                   info = None,
                                   bbox_format = None,
                                   **kwargs):
    """Annotates bounding boxes onto an image.

    This method will annotate the given bounding boxes directly onto the image,
    with optional category labels also enabling class-specific text and coloring
    to be annotated above the bounding box. Bounding boxes can be either in COCO
    JSON format or a list of 4-tuples.

    Parameters
    ----------
    image : Any
        Either the image, or a path to the image.
    bboxes : Any
        Either a COCO JSON dictionary with bounding boxes and labels, or a list
        of 4-tuples with bounding box coordinates.
    segmentation : Any
        A list of polygonal segmentation masks. This can either be in the standard
        COCO JSON format (e.g., two lists, each containing a list of coordinates),
        or it can be a processed coordinate array.
    labels : Any
        An optional list of labels (if `bboxes` is a list of 4-tuples).
    inplace : bool
        Whether to modify the image in-place.
    info : Any
        An optional list of additional information to aid in annotating labels for
        the bounding boxes. If this is not passed but a list of labels is, then the
        bounding box will simply be displayed with a changed color, not the numerical
        label. However, if this is passed, then the displayed label will be the
        corresponding value in `info`. This being said, if `labels` is a list of
        strings, then that will be used instead.
    bbox_format : str
        The format of the bounding boxes. If this is not passed, then the bounding
        boxes will be assumed to be in COCO JSON format.
    kwargs : dict
        Keyword arguments to pass to `cv2.rectangle`.

    Returns
    -------
    An image with annotated bounding boxes.
    """
    # Check if everything has been passed to the first argument (e.g., if the
    # user just called `agml.viz.put_boxes_on_image(loader[0])`), or whether
    # the relevant values have been passed to the keyword arguments.
    if bboxes is None and labels is None and segmentation is None:
        if len(image) == 4: # all values passed to the first argument
            image, bboxes, labels, segmentation = image
        if len(image) == 2: # COCO JSON dict
            image, bboxes = image
    elif bboxes is not None and labels is None and segmentation is None:
        if len(bboxes) == 3: # all annotations passed to second argument
            bboxes, labels, segmentation = bboxes
        if isinstance(bboxes, dict): # COCO JSON dict
            pass
    else:
        # If there is another form of annotation passed, then check that
        # at least `segmentation` exists (it's less relevant that we have
        # `bboxes` and `labels`).
        if segmentation is None:
            raise ValueError(
                "You need to pass some form of instance segmentation mask to "
                "`annotate_instance_segmentation`, instead got no mask")

    # If the user passed a COCO JSON dictionary for `bboxes`, resolve it into
    # a list of bounding boxes, labels, and the segmentation mask.
    if isinstance(bboxes, dict):
        try:
            bboxes, labels, segmentation = \
                bboxes['bbox'], bboxes['category_id'], bboxes['segmentation']
        except KeyError:
            try:
                bboxes, labels, segmentation = \
                    bboxes['bboxes'], bboxes['labels'], bboxes['segmentation']
            except KeyError:
                raise ValueError(
                    "Unexpected COCO JSON format found in input `bboxes` "
                    "dictionary, got {list(bboxes.keys())} but expected "
                    "either `bbox` or `bboxes` for bounding boxes.")
    if bbox_format is not None:
        bboxes = convert_bbox_format(bboxes, bbox_format)

    # Format the instance segmentation polygon masks into the correct format.


    # Run a few final checks in order to ensure data is formatted properly.
    image = format_image(image, mask = False)
    if not inplace:
        image = image.copy()
    bboxes = weak_squeeze(bboxes, ndims = 2)
    labels = weak_squeeze(labels, ndims = 1)

    # Check for any additional information that can be used to annotate labels.
    annotate_with_info = False
    if info is not None:
        if all(isinstance(label, str) for label in labels):
            # If the labels are already strings, then create an info dictionary
            # that maps each label to a specific integer value, then reverse it.
            annotate_with_info = True
            unique_labels = np.unique(labels)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            info = {v: k for k, v in label_map.items()}
            labels = [label_map[j] for j in labels]

        elif hasattr(info, 'num_to_class'):
            annotate_with_info = True
            info = info.num_to_class

        elif isinstance(info, dict):
            annotate_with_info = True

    # Check the keyword arguments for any additional information that can be used.
    thickness = kwargs.get('thickness', 2)

    # Iterate over each bounding box and label, and annotate them onto the image.
    cmap = get_colormap()
    shape = image.shape[:2]
    for bbox, label in zip(bboxes, labels):
        # Scale the bounding box if necessary.
        bbox = _resolve_proportional_bboxes(bbox, shape)
        x, y = bbox[:2]
        x2, y2 = bbox[2] + x, bbox[3] + y

        # Annotate the bounding box onto the image.
        cv2.rectangle(image, (x, y), (x2, y2), color = cmap[as_scalar(label)],
                      thickness = thickness)

        # If the user passed additional information, annotate it onto the image
        # by putting the text above the bounding box with the corresponding label.
        if annotate_with_info:
            text = info[label]

            # Get the text size.
            (label_width, label_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Calculate the coordinates of the background rectangle of the label.
            x, y = bbox[0], bbox[1] - label_height - baseline
            x2, y2 = x + label_width, y + label_height + baseline

            # Annotate the background rectangle and label text onto the image.
            cv2.rectangle(image, (x, y), (x2, y2), color = cmap[label], thickness = -1)
            cv2.putText(image, text, (x, y + label_height),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    # Return the image.
    return image