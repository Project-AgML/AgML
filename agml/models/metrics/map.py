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

import torch
import torch.nn as nn

from agml.utils.logging import log


@torch.jit.script
def bbox_iou(predicted_box, truth_box):
    """Calculates the IOU of predicted and truth bounding boxes."""
    # Calculate the coordinates of the bounding box overlap.
    box1_x1 = predicted_box[..., 0:1]
    box1_y1 = predicted_box[..., 1:2]
    box1_x2 = predicted_box[..., 0:1] + predicted_box[..., 2:3]
    box1_y2 = predicted_box[..., 1:2] + predicted_box[..., 3:4]
    box2_x1 = truth_box[..., 0:1]
    box2_y1 = truth_box[..., 1:2]
    box2_x2 = truth_box[..., 0:1] + truth_box[..., 2:3]
    box2_y2 = truth_box[..., 1:2] + truth_box[..., 3:4]
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Get the area of the union.
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union_area = (box1_area + box2_area - intersection)

    # Return the intersection over union.
    return intersection / (union_area + 1e-6)


@torch.no_grad()
def mean_average_precision(
        predicted_boxes, truth_boxes, num_classes = 1, iou_thresh = 0.5):
    """Calculates the mean average precision for predicted and true boxes."""
    average_precisions = []

    # Check whether to add confidence scores to the ground truth boxes.
    if len(truth_boxes[0]) == 6:
        out_boxes = []
        for a in truth_boxes:
            out_boxes.append([*a[:2], 1.0, *a[2:]])
        truth_boxes = out_boxes.copy()

    # Calculate average precision for each class.
    pred_boxes, true_boxes = torch.tensor(predicted_boxes), torch.tensor(truth_boxes)
    for c in range(num_classes):
        # Get the predictions and targets corresponding to this class.
        detections = pred_boxes[torch.where(pred_boxes[:, 1] == c)[0]].tolist()
        ground_truths = true_boxes[torch.where(true_boxes[:, 1] == c)[0]].tolist()

        # If there are no ground truths, then the per-class AP is 0.
        if len(ground_truths) == 0:
            average_precisions.append(0.0)
            continue

        # Get all of the unique data samples and create a dictionary
        # storing all of the corresponding bounding boxes for each sample.
        training_ids = torch.unique(true_boxes[:, 0])
        truth_samples_by_id = {
            idx.numpy().item(): true_boxes[torch.where(true_boxes[:, 0] == idx)] # noqa
            for idx in training_ids}

        # Determine the number of boxes for each of the training samples.
        numpy_gt = torch.tensor(ground_truths)
        amount_bboxes = {int(k.numpy().item()): torch.zeros(v) for k, v in zip(
            *torch.unique(numpy_gt[:, 0], return_counts = True))}

        # Sort the boxes by probabilities.
        detections.sort(key = lambda x: x[2], reverse = True)
        tp = torch.zeros((len(detections)))
        fp = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If there are no boxes for this class, then there are
        # no calculations to do, so skip it.
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Get the update number for this detection.
            update_num = int(detection[0])

            # Only take out the ground_truths that have the same
            # training idx as the detection.
            ground_truth_img = truth_samples_by_id[update_num]

            # Get the bounding box with the highest IoU.
            ious = torch.tensor([bbox_iou(
                torch.tensor(detection[3:]), gt[3:].clone())
                for gt in ground_truth_img])
            best_iou = torch.max(ious)
            best_gt_idx = torch.argmax(ious)

            # If the IoU is above the threshold, then it may be a true positive.'
            if best_iou > iou_thresh:
                # This should be the first time the box is detected. Otherwise,
                # that would mean that there are multiple predicted bounding
                # boxes for the same object, which is a false positive.
                try:
                    if amount_bboxes[update_num][best_gt_idx] == 0:
                        tp[detection_idx] = 1
                        amount_bboxes[update_num][best_gt_idx] = 1
                    else:
                        fp[detection_idx] = 1
                except KeyError:
                    # A false detection.
                    fp[detection_idx] = 1

            # If the IoU is below the threshold, then it is a false positive.
            else:
                fp[detection_idx] = 1

        # Calculate the prediction/recalls and update the array.
        tp_cumsum = torch.cumsum(tp, dim = 0)
        recalls = tp_cumsum / (total_true_bboxes + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + torch.cumsum(fp, dim = 0) + 1e-6)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    # Calculate the mean of all of the average precisions.
    return sum(average_precisions) / len(average_precisions)


class MeanAveragePrecision(nn.Module):
    """A metric to calculate mean average precision for object detection tasks.

    This class is used as a wrapper around mean average precision calculations,
    which allows for accumulation of predictions over time. The `update` (and
    `batch_update`) methods can be used to update data in the class, then `compute`
    to get the calculated mean average precision, and finally `reset` to restore
    the accumulators to an empty state, allowing from-scratch calculations.
    """

    def __init__(self, num_classes = 1, iou_threshold = 0.5):
        # Set base parameters.
        super(MeanAveragePrecision, self).__init__()
        self._num_classes = num_classes
        self._iou_threshold = iou_threshold

        # Store the truth and prediction data in containers.
        self._prediction_data, self._truth_data = [], []
        self._num_updates = 0

    @staticmethod
    def _scalar_to_array(*args):
        """Converts 0-dimensional scalar arrays to 1-d arrays."""
        cvt = lambda x: np.expand_dims(x, 0) if x.ndim == 0 else x
        outs = [cvt(arg) for arg in args]
        return outs[0] if len(args) == 1 else outs

    def update(self, pred_data, gt_data):
        """Update the tracker with prediction and ground truth data.

        The arguments `pred_data` and `gt_data` should be either dictionaries
        (with the following keys), or lists of values which correspond in order
        to the same keys listed below:

        - `pred_data`: `boxes`, `labels`, and `scores`.
        - `gt_data`: `boxes` and `labels`.

        Note: To update a batch of data, use `batch_update()`.
        """
        # Get the relevant data from the input arguments.
        if isinstance(pred_data, dict):
            pred_boxes, pred_labels, pred_scores = \
                pred_data['boxes'], pred_data['labels'], pred_data['scores']
        else:
            pred_boxes, pred_labels, pred_scores = pred_data
        if isinstance(gt_data, dict):
            gt_boxes, gt_labels = \
                gt_data['boxes'], gt_data['labels']
        else:
            gt_boxes, gt_labels = gt_data

        # Format the data.
        pred_boxes = np.squeeze(pred_boxes)
        pred_labels = np.squeeze(pred_labels)
        pred_scores = np.squeeze(pred_scores)
        pred_labels, gt_labels, pred_scores = \
            self._scalar_to_array(pred_labels, gt_labels, pred_scores)
        if pred_boxes.ndim == 1:
            pred_boxes = np.expand_dims(pred_boxes, axis = 0)
        if gt_boxes.ndim == 1:
            gt_boxes = np.expand_dims(gt_boxes, axis = 0)

        # Create the data in the proper format.
        for bbox, label in zip(gt_boxes, gt_labels):
            self._truth_data.append(
                [self._num_updates, int(label - 1), 1.0, *bbox])
        if pred_boxes.ndim == 1:
            pred_boxes = np.expand_dims(pred_boxes, axis = 0)
        for bbox, label, score in zip(pred_boxes, pred_labels, pred_scores):
            self._prediction_data.append(
                [self._num_updates, int(label - 1), score, *bbox])

        # Increment the number of updates.
        self._num_updates += 1

    def compute(self):
        """Computes the mean average precision with the given data."""
        if self._num_updates == 0:
            log("Tried to compute mean average precision "
                "without any data updates; returning 0.0.")
            return 0.0

        return mean_average_precision(
            predicted_boxes = self._prediction_data,
            truth_boxes = self._truth_data,
            num_classes = self._num_classes,
            iou_thresh = self._iou_threshold,
        )

    def reset(self):
        """Resets the mean average precision."""
        del self._prediction_data, self._truth_data
        self._prediction_data, self._truth_data = [], []
        self._num_updates = 0





