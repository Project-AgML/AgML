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
import matplotlib.pyplot as plt

from agml.utils.general import (
    resolve_tuple_values, weak_squeeze, scalar_unpack, as_scalar
)
from agml.data.tools import convert_bbox_format
from agml.viz.tools import format_image, get_colormap, convert_figure_to_image
from agml.viz.display import display_image


def _resolve_proportional_bboxes(coords, shape):
    """Resolves float vs. integer bounding boxes."""
    coords = scalar_unpack(coords)

    # If the coordinates are all float values, then scale them up to the image
    # size (if they are less than 1) or simply return integer values otherwise.
    if all(isinstance(i, float) for i in coords):
        if coords[0] <= 1:
            x, y, width, height = coords
            y *= shape[0]; height *= shape[0]  # noqa
            x *= shape[1]; width *= shape[1]  # noqa
            coords = [x, y, width, height]
        return [int(round(c)) for c in coords]
    elif all(isinstance(i, int) for i in coords):
        return coords
    raise TypeError(
        f"Got multiple types for coordinates: "
        f"{[type(i) for i in coords]}.")


def annotate_object_detection(image,
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
    image, bboxes, labels = resolve_tuple_values(
        image, bboxes, labels, custom_error =
        "If `image` is a tuple/list, it should contain "
        "three values: the image, mask, and (optionally) labels."
    )

    # If the user passed a COCO JSON dictionary for `bboxes`, resolve it into
    # a list of bounding boxes and its corresponding labels.
    if isinstance(bboxes, dict):
        try:
            bboxes, labels = bboxes['bbox'], bboxes['category_id']
        except KeyError:
            try:
                bboxes, labels = bboxes['bboxes'], bboxes['labels']
            except KeyError:
                raise ValueError(
                    "Unexpected COCO JSON format found in input `bboxes` "
                    "dictionary, got {list(bboxes.keys())} but expected "
                    "either `bbox` or `bboxes` for bounding boxes.")

    # If there are no bounding boxes, then simply return the image.
    if len(bboxes) == 0:
        return image

    # Format the bounding boxes and labels.
    if bbox_format is not None:
        bboxes = convert_bbox_format(bboxes, bbox_format)

    # Run a few final checks in order to ensure data is formatted properly.
    image = format_image(image, mask = False).copy()
    if not inplace:
        image = image.copy()
    bboxes = weak_squeeze(bboxes, ndims = 2)
    if labels is None:
        labels = [0] * len(bboxes)
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


def show_image_and_boxes(image,
                         bboxes = None,
                         labels = None,
                         inplace = True,
                         info = None,
                         bbox_format = None,
                         **kwargs):
    """Visualizes an image with annotated bounding boxes."""
    # Annotate the bounding boxes onto the image.
    image = annotate_object_detection(image = image,
                                      bboxes = bboxes,
                                      labels = labels,
                                      inplace = inplace,
                                      info = info,
                                      bbox_format = bbox_format,
                                      **kwargs)

    # Display the image.
    _ = display_image(image, matplotlib_figure = False)
    return image


def show_object_detection_truth_and_prediction(image,
                                               real_boxes = None,
                                               real_labels = None,
                                               predicted_boxes = None,
                                               predicted_labels = None,
                                               inplace = True,
                                               info = None,
                                               bbox_format = None,
                                               **kwargs):
    """Visualizes the ground truth and prediction of an object detection model.

    Parameters
    ----------
    image : array-like
        The image to visualize.
    real_boxes : array-like, optional
        The real bounding boxes of the image.
    real_labels : array-like, optional
        The real labels of the image.
    predicted_boxes : array-like, optional
        The predicted bounding boxes of the image.
    predicted_labels : array-like, optional
        The predicted labels of the image.
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
    The modified image. If you don't want to display the output, then pass
    the optional keyword argument `no_show`.
    """
    # Parse the inputs. These are the following possible formats:
    #
    # (1) Five separate inputs (e.g., the image, real boxes, real labels, predicted
    #     boxes, and predicted labels). These can also all be condensed into a single
    #     argument, which would be the first argument, `image`.
    # (2) Three separate inputs (e.g., the image, a COCO JSON dictionary with real
    #     annotations, and a COCO JSON dictionary with predicted annotations). These
    #     can also all be condensed into a single argument, which would be `image`.
    # (3) Four unique inputs (this would be a combination of the first two cases, where
    #     one format has been passed for real annotations and another for predicted
    #     annotations, and we need to sift for this in the input arguments.
    #
    # The following checks for all these cases (though not in this specific order).
    if all(arg is None for arg in (real_boxes, real_labels, predicted_boxes, predicted_labels)):
        # Check for the case when all arguments are condensed.
        if len(image) == 3:
            image, real_boxes, predicted_boxes = image
        if len(image) == 5:
            image, real_boxes, real_labels, predicted_boxes, predicted_labels = image
    else:
        # Otherwise, check that the first three arguments have been passed.
        if real_boxes is not None and real_labels is not None and \
                predicted_boxes is None and predicted_labels is None:
            if isinstance(real_boxes, dict) and isinstance(real_labels, dict):
                predicted_boxes = real_labels
                real_labels = None

        # Otherwise, check that the first FOUR arguments have been passed.
        if real_boxes is not None and real_labels is not None and \
                predicted_boxes is not None and predicted_labels is None:
            # The real annotations are a COCO JSON dict and predicted annotations
            # are two input arrays. The other case is redundant since it is technically
            # correct in formatting, so we don't do anything.
            if isinstance(real_boxes, dict) and not isinstance(real_labels, dict):
                predicted_labels = predicted_boxes
                predicted_boxes = real_labels
                real_labels = None

        # Otherwise, just roll with the given format and see if it works.
        else:
            pass

    # Generate the two images: real and predicted.
    real_image = annotate_object_detection(image = image,
                                           bboxes = real_boxes,
                                           labels = real_labels,
                                           inplace = inplace,
                                           info = info,
                                           bbox_format = bbox_format,
                                           **kwargs)
    predicted_image = annotate_object_detection(image = image,
                                                bboxes = predicted_boxes,
                                                labels = predicted_labels,
                                                inplace = inplace,
                                                info = info,
                                                bbox_format = bbox_format,
                                                **kwargs)


    # Create two side-by-side figures with the images.
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    for ax, img, label in zip(
            axes, (real_image, predicted_image),
            ("Ground Truth Boxes", "Predicted Boxes")):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(label, fontsize = 15)
        ax.set_aspect('equal')
    fig.tight_layout()

    # Display and return the image.
    image = convert_figure_to_image()
    if not kwargs.get('no_show', False):
        _ = display_image(image)
    return image


