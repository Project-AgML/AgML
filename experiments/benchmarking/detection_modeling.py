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

import os
from typing import Union

import numpy as np
import torch
from effdet import DetBenchTrain, create_model_from_config, get_efficientdet_config

from agml.models.detection import DetectionModel
from experiments.benchmarking.mean_average_precision_torch import MeanAveragePrecision


class DetectionTrainingModel(DetectionModel):
    """Wraps an `EfficientDet` model for a training experiment."""

    def __init__(
        self,
        num_classes: int = 1,
        pretrained_weights: str = None,
        confidence_threshold: float = 0.3,
        learning_rate: float = 0.0002,
        wbf_iou_threshold: float = 0.44,
        **kwargs,
    ):
        # Initialize the super module.
        super(DetectionTrainingModel, self).__init__(model_initialized=True)

        # Initialize the model using the provided arguments for customizability.
        self.model = self.make_model(
            num_classes=num_classes,
            pretrained_weights=pretrained_weights,
            architecture=kwargs.get("architecture", "tf_efficientdet_d4"),
        )

        # Set the training parameters.
        self._confidence_threshold = confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold

        # Construct the metric.
        self.val_dataset_adaptor = None
        self.map = MeanAveragePrecision(self.model.num_classes)
        self._sanity_check_passed = False

    @staticmethod
    def make_model(
        num_classes: int,
        pretrained_weights: str,
        image_size: Union[int, tuple] = 512,
        architecture: str = "tf_efficientdet_d4",
    ):
        """Constructs the `EfficientDet` model from the provided parameters."""
        # Parse the input arguments.
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if pretrained_weights is None or pretrained_weights is False:
            pretrained_weights = ""

        # Construct the configuration.
        cfg = get_efficientdet_config(architecture)
        cfg.update({"image_size": image_size})

        # Build the model.
        model_default_pretrained = False
        if pretrained_weights == "coco":
            model_default_pretrained = True
        net = create_model_from_config(
            cfg, pretrained=model_default_pretrained, num_classes=num_classes
        )

        # Load the pretrained weights if they are provided.
        if os.path.exists(pretrained_weights):
            # Auto-inference the number of classes from the state dict.
            state = torch.load(pretrained_weights, map_location="cpu")
            weight_key = [
                i for i in state.keys() if "class_net.predict.conv_pw.weight" in i
            ][0]
            pretrained_num_classes = int(state[weight_key].shape[0] / 9)

            # Load the pretrained weights.
            net.reset_head(num_classes=pretrained_num_classes)
            net.load_state_dict(state)

            # Restore the number of classes.
            if num_classes != pretrained_num_classes:
                net.reset_head(num_classes=num_classes)

        # Return the network.
        return DetBenchTrain(net)

    @staticmethod
    def _rescale_bboxes_xyxy(predicted_bboxes, image_sizes):
        """Re-scales output bounding boxes to the original image sizes.

        This is re-written in this training subclass since annotations here are
        also mapped to the YXYX format, and that needs to be accounted for.
        """
        scaled_boxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            h, w = img_dims
            if len(bboxes) > 0:
                # Re-scale the bounding box to the appropriate format.
                scale_ratio = [h / 512, w / 512, h / 512, w / 512]
                scaled = (np.array(bboxes.detach().cpu()) * scale_ratio).astype(
                    np.int32
                )

                # Convert the Pascal-VOC (yxyx) format to COCO (xywh).
                y, x = scaled[:, 0], scaled[:, 1]
                h, w = scaled[:, 2] - y, scaled[:, 3] - x
                scaled_boxes.append(np.dstack((x, y, w, h)))
                continue

            # Otherwise, there is no prediction for this image.
            scaled_boxes.append(np.array([]))
        return scaled_boxes

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # Run a forward pass through the model.
        images, annotations, _ = batch
        losses = self.model(images, annotations)

        # Calculate and log losses.
        self.log(
            "train_loss",
            losses["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_class_loss",
            losses["class_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "train_box_loss",
            losses["box_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return losses["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets = batch
        outputs = self.model(images, annotations)
        detections = outputs["detections"]

        # Calculate the mean average precision.
        if self._sanity_check_passed:
            boxes, confidences, labels = self._process_detections(
                self._to_out(detections)
            )
            boxes = self._rescale_bboxes(boxes, [[512, 512]] * len(images))
            annotations["bbox"] = self._rescale_bboxes_xyxy(
                annotations["bbox"],
                [
                    [
                        512,
                        512,
                    ]
                ]
                * len(images),
            )

            for pred_box, pred_label, pred_conf, true_box, true_label in zip(
                boxes, labels, confidences, annotations["bbox"], annotations["cls"]
            ):
                metric_update_values = (
                    dict(
                        boxes=self._to_out(torch.tensor(pred_box, dtype=torch.float32)),
                        labels=self._to_out(
                            torch.tensor(pred_label, dtype=torch.int32)
                        ),
                        scores=self._to_out(torch.tensor(pred_conf)),
                    ),
                    dict(
                        boxes=self._to_out(torch.tensor(true_box, dtype=torch.float32)),
                        labels=self._to_out(
                            torch.tensor(true_label, dtype=torch.int32)
                        ),
                    ),
                )
                self.map.update(*metric_update_values)

                # Log the MAP values.
                map_ = self.map.compute().detach().cpu().numpy().item()
                self.log(
                    "map",
                    map_,
                    prog_bar=True,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log(
            "valid_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "valid_class_loss",
            logging_losses["class_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "valid_box_loss",
            logging_losses["box_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": outputs["loss"], "batch_predictions": batch_predictions}

    def on_validation_epoch_end(self) -> None:
        """Log mean average precision at the end of each epoch."""
        # No validation should be run during the sanity check step.
        if not self._sanity_check_passed:
            self._sanity_check_passed = True
            return

        # Compute the mean average precision and reset it.
        if hasattr(self, "map"):
            map_ = self.map.compute().detach().cpu().numpy().item()
            self.log(
                "map_epoch",
                map_,
                prog_bar=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.map.reset()

    def get_progress_bar_dict(self) -> dict:
        """Remove the `v_num` from the bar; it takes away valuable space."""
        p_bar = super(DetectionTrainingModel, self).get_progress_bar_dict()
        p_bar.pop("v_num", None)
        return p_bar
