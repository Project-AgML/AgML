import numpy as np
import torch
from tqdm import tqdm


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


from collections import Counter

import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # print(intersection, box1_area, box2_area, boxes_preds, boxes_labels)
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20,
    use_bar=False,
):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    if len(true_boxes[0]) == 6:
        my_new_boxes = true_boxes.copy()
        _add_truth_scores_to_annotations(my_new_boxes)
        true_boxes = my_new_boxes

    # used for numerical stability later on
    epsilon = 1e-6

    classes = range(num_classes)
    if use_bar:
        classes = tqdm(classes, leave=False)

    for c in classes:
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    if len(average_precisions) == 0:
        return torch.tensor(0.0)
    return sum(average_precisions) / len(average_precisions)


class MeanAveragePrecision(object):
    """Stores and calculates the mean average precision over data."""

    def __init__(self, num_classes=1, use_bar=False):
        self._num_classes = num_classes

        # Store all of the ground truth and prediction data.
        self._prediction_data = []
        self._ground_truth_data = []

        # Track the number of times that a data sample is added,
        # and any past MAP values which are computed.
        self._num_updates = 0
        self._prior_maps = {}

        # Whether to use a progress bar.
        self._use_bar = use_bar

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
            use_bar=self._use_bar,
        )
        self._prior_maps[self._num_updates] = ap
        return ap

    def reset(self):
        """Resets the internal lists."""
        del self._prediction_data
        del self._ground_truth_data
        self._prediction_data = []
        self._ground_truth_data = []
        self._num_updates = 0
