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


def _scalar_to_array(*args):
    """Converts 0-dimensional scalar arrays to 1-d arrays."""
    cvt = lambda x: np.expand_dims(x, 0) if x.ndim == 0 else x
    outs = [cvt(arg) for arg in args]
    return outs[0] if len(args) == 1 else outs


def _add_truth_scores_to_annotations(annotations):
    """Adds a `score` element of 1.0 to ground truth annotations.

    As for the mean average precision method, boxes are expected to be
    in the format [image_idx, class, *score*, x1, y1, x2, y2], but
    ground truth elements do not have a score, this method takes
    elements of the form [image_idx, class, x1, y1, x2, y2], and adds
    a dummy `score` element of 1.0 to satisfy the MAP method.
    """
    a = annotations  # short-hand
    for idx in range(len(annotations)):
        prior, after = a[idx][:2], a[idx][2:]
        a[idx] = [*prior, 1.0, *after]


def intersection_over_union(pred_boxes, gt_boxes):
    """Calculates intersection over union between predicted/target boxes.

    This method expects bounding boxes in the pascal-voc format:
    (x1, y1, x2, y2), and they should be of shape (num_boxes, 4).
    """
    # Determine the coordinates of the overlapping square.
    x1 = np.maximum(pred_boxes[..., 0:1], gt_boxes[..., 0:1])
    y1 = np.maximum(pred_boxes[..., 1:2], gt_boxes[..., 1:2])
    x2 = np.minimum(pred_boxes[..., 2:3], gt_boxes[..., 2:3])
    y2 = np.minimum(pred_boxes[..., 3:4], gt_boxes[..., 3:4])

    # Find the area of the two boxes and calculate the IoU.
    intersection = (x2 - x1).clip(0, None) * (y2 - y1).clip(0, None)
    pred_area = abs(
        (pred_boxes[..., 2:3] - pred_boxes[..., 0:1])
        * (pred_boxes[..., 3:4] - pred_boxes[..., 1:2])
    )
    gt_area = abs(
        (gt_boxes[..., 2:3] - gt_boxes[..., 0:1])
        * (gt_boxes[..., 3:4] - gt_boxes[..., 1:2])
    )
    return intersection / (pred_area + gt_area - intersection + 1e-6)


def mean_average_precision(pred_boxes, true_boxes, num_classes=1, iou_threshold=0.5):
    """Calculates the mean average precision (mAP) of data samples."""
    # Since we modify the input arguments in-place, make a copy so as
    # not to affect the original variables which are passed to this method.
    pred_boxes = pred_boxes.copy()
    true_boxes = true_boxes.copy()

    # Check whether to add confidence scores to the ground truth boxes.
    if len(true_boxes[0]) == 6:
        _add_truth_scores_to_annotations(true_boxes)

    # Track the average precision for each of the objects.
    average_precisions = []

    for c in range(num_classes):
        # Get the corresponding predictions and targets for this class.
        where_pred_box = np.array(pred_boxes)
        where_true_box = np.array(true_boxes)
        detections = where_pred_box[np.where(where_pred_box[:, 1] == c)[0]].tolist()
        ground_truths = where_true_box[np.where(where_true_box[:, 1] == c)[0]].tolist()

        # Get all of the unique data samples and create a dictionary
        # storing all of the corresponding bounding boxes for each sample.
        training_ids = np.unique(where_true_box[:, 0])
        truth_samples_by_id = {
            idx: where_true_box[np.where(where_true_box[:, 0] == idx)]
            for idx in training_ids
        }

        # Determine the number of boxes for each of the training samples
        numpy_gt = np.array(ground_truths)
        counts = np.bincount(numpy_gt[:, 0].astype(np.int64))
        indices = np.nonzero(counts)[0]
        amount_bboxes = {k: np.zeros(v) for k, v in zip(indices, counts)}

        # Sort the boxes by probabilities.
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = np.zeros((len(detections)))
        FP = np.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If there are no boxes for this class, then there are
        # no calculations to do, so skip it.
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = truth_samples_by_id[int(detection[0])]

            # Get the bounding box with the highest IoU.
            ious = np.array(
                [
                    intersection_over_union(np.array(detection[3:]), np.array(gt[3:]))
                    for gt in ground_truth_img
                ]
            )
            best_iou = np.max(ious)
            best_gt_idx = np.argmax(ious)

            # If the IoU is above the threshold, then it may be a true positive.
            if best_iou > iou_threshold:
                # This should be the first time the box is detected. Otherwise,
                # that would mean that there are multiple predicted bounding
                # boxes for the same object, which is a false positive.
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # If the IoU is below the threshold, then it is a false positive.
            else:
                FP[detection_idx] = 1

        # Calculate the prediction/recalls and update the array.
        tp_cumsum = np.cumsum(TP, axis=0)
        recalls = tp_cumsum / (total_true_bboxes + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + np.cumsum(FP, axis=0) + 1e-6)
        precisions = np.concatenate((np.array([1]), precisions))
        recalls = np.concatenate((np.array([0]), recalls))
        average_precisions.append(np.trapz(precisions, recalls))

    # Calculate the mean of all of the average precisions.
    return sum(average_precisions) / len(average_precisions)


class MeanAveragePrecision(object):
    """Stores and calculates the mean average precision over data."""

    def __init__(self, num_classes=1):
        self._num_classes = num_classes

        # Store all of the ground truth and prediction data.
        self._prediction_data = []
        self._ground_truth_data = []

        # Track the number of times that a data sample is added,
        # and any past MAP values which are computed.
        self._num_updates = 0
        self._prior_maps = {}

    def batch_update(self, pred_data, gt_data):
        """Same as `update()`, but for batches of data."""
        for pred_d, gt_d in zip(pred_data, gt_data):
            self.update(pred_d, gt_d)

    def update(self, pred_data, gt_data):
        """Update the tracker with prediction and ground truth data.

        The arguments `pred_data` and `gt_data` should be either
        dictionaries (with the following keys), or lists of values
        which correspond in order to the same keys listed below:

        - `pred_data`: `boxes`, `labels`, and `scores`.
        - `gt_data`: `boxes` and `labels`.

        Note: To update a batch of data, use `batch_update()`.
        """
        # Get the relevant data from the input arguments.
        if isinstance(pred_data, dict):
            pred_boxes, pred_labels, pred_scores = (
                pred_data["boxes"],
                pred_data["labels"],
                pred_data["scores"],
            )
        else:
            pred_boxes, pred_labels, pred_scores = pred_data

        if isinstance(gt_data, dict):
            gt_boxes, gt_labels = gt_data["boxes"], gt_data["labels"]
        else:
            gt_boxes, gt_labels = gt_data

        # Format the data.
        pred_boxes = np.squeeze(pred_boxes)
        pred_labels = np.squeeze(pred_labels)
        pred_scores = np.squeeze(pred_scores)
        pred_labels, gt_labels, pred_scores = _scalar_to_array(
            pred_labels, gt_labels, pred_scores
        )
        if pred_boxes.ndim == 1:
            pred_boxes = np.expand_dims(pred_boxes, axis=0)
        if gt_boxes.ndim == 1:
            gt_boxes = np.expand_dims(gt_boxes, axis=0)

        # Create the data in the proper format.
        for bbox, label in zip(gt_boxes, gt_labels):
            self._ground_truth_data.append([self._num_updates, int(label - 1), *bbox])
        if pred_boxes.ndim == 1:
            pred_boxes = np.expand_dims(pred_boxes, axis=0)
        for bbox, label, score in zip(pred_boxes, pred_labels, pred_scores):
            self._prediction_data.append(
                [self._num_updates, int(label - 1), score, *bbox]
            )

        # Increment the update number.
        self._num_updates += 1

    @property
    def historical_data(self):
        """Returns historical mAP over past data."""
        return self._prior_maps

    def compute(self, iou_threshold=0.5):
        """Computes the mAP with the data."""
        ap = mean_average_precision(
            self._prediction_data,
            self._ground_truth_data,
            num_classes=self._num_classes,
            iou_threshold=iou_threshold,
        )
        self._prior_maps[self._num_updates] = ap
        return ap
