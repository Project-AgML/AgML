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

"""
Some of the training code in this file is adapted from the following sources:

1. https://github.com/rwightman/efficientdet-pytorch
2. https://gist.github.com/Chris-hughes10/73628b1d8d6fc7d359b3dcbbbb8869d7
"""

import argparse
import os
import warnings
from typing import List, Union

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from effdet import DetBenchTrain, EfficientDet, create_model_from_config, get_efficientdet_config
from effdet.efficientdet import HeadNet
from ensemble_boxes import ensemble_boxes_wbf
from mean_average_precision_torch import MeanAveragePrecision as MAP
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from tools import MetricLogger, auto_move_data, checkpoint_dir, gpus
from torch.utils.data import DataLoader, Dataset

import agml

# Constants
IMAGE_SIZE = 512


def create_model(num_classes=1, architecture="tf_efficientdet_d4", pretrained=False):
    config = get_efficientdet_config(architecture)
    config.update({"image_size": (IMAGE_SIZE, IMAGE_SIZE)})

    print(config)

    net = create_model_from_config(
        config, pretrained=pretrained, num_classes=num_classes
    )
    net.class_net = HeadNet(
        config,
        num_outputs=num_classes,
    )
    return DetBenchTrain(net, config)


class AgMLDatasetAdaptor(object):
    """Adapts an AgML dataset for use in a `LightningDataModule`."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def get_image_and_labels_by_idx(self, index):
        image, annotation = self.loader[index]
        image = Image.fromarray(image)
        bboxes = np.array(annotation["bbox"]).astype(np.int32)
        x_min = bboxes[:, 0]
        y_min = bboxes[:, 1]
        x_max = bboxes[:, 2] + x_min
        y_max = bboxes[:, 3] + y_min
        x_min, y_min = np.clip(x_min, 0, image.width), np.clip(y_min, 0, image.height)
        x_max, y_max = np.clip(x_max, 0, image.width), np.clip(y_max, 0, image.height)
        bboxes = np.dstack((x_min, y_min, x_max, y_max)).squeeze(axis=0)
        class_labels = np.array(annotation["category_id"]).squeeze()
        return image, bboxes, class_labels, index


def get_transforms(mode="inference"):
    """Returns a set of transforms corresponding to the mode."""
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1),
                ToTensorV2(p=1),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0,
                min_visibility=0,
                label_fields=["labels"],
            ),
        )

    elif mode in ["val", "validation"]:
        return A.Compose(
            [A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1), ToTensorV2(p=1)],
            p=1.0,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0,
                min_visibility=0,
                label_fields=["labels"],
            ),
        )

    elif mode == "inference":
        return A.Compose(
            [A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1), ToTensorV2(p=1)], p=1.0
        )


class EfficientDetDataset(Dataset):
    def __init__(self, adaptor, transforms=None):
        self.ds = adaptor
        if transforms is None:
            transforms = get_transforms("val")
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image, pascal_bboxes, class_labels, image_id = (
            self.ds.get_image_and_labels_by_idx(index)
        )

        # Add a label dimension for consistency.
        if class_labels.ndim == 0:
            class_labels = np.expand_dims(class_labels, axis=0)

        # Construct the sample.
        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        try:
            sample = self.transforms(**sample)
        except:  # debugging
            raise Exception(f"Failed sample: {sample}")

        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        labels = sample["labels"]

        # Convert 1-channel and 4-channel to 3-channel.
        if image.shape[0] == 1:
            image = torch.tile(image, (3, 1, 1))
        if image.shape[0] == 4:
            image = image[:3]

        # Convert to yxyx from xyxy.
        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]

        # Create the target from the annotations.
        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }
        return image, target, image_id


class EfficientDetDataModule(pl.LightningDataModule):
    """A `LightningDataModule` for the `LightningModule`."""

    def __init__(
        self,
        train_dataset_adaptor,
        validation_dataset_adaptor,
        train_transforms=None,
        val_transforms=None,
        num_workers=4,
        batch_size=8,
    ):
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        if train_transforms is None:
            train_transforms = get_transforms("train")
        self.train_tfms = train_transforms
        if val_transforms is None:
            val_transforms = get_transforms("val")
        self.val_tfms = val_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(adaptor=self.train_ds, transforms=self.train_tfms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(adaptor=self.valid_ds, transforms=self.val_tfms)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }
        return images, annotations, targets, image_ids


class EfficientDetModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=1,
        confidence_threshold=0.3,
        learning_rate=0.0002,
        wbf_iou_threshold=0.44,
        inference_transforms=None,
        architecture="efficientdet_d4",
        save_dir=None,
        pretrained=False,
        validation_dataset_adaptor=None,
    ):
        super().__init__()
        self.model = create_model(
            num_classes, architecture=architecture, pretrained=pretrained
        )
        self.confidence_threshold = confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        if inference_transforms is None:
            inference_transforms = get_transforms("inference")
        self.inference_tfms = inference_transforms

        # Construct the metric.
        self.val_dataset_adaptor = None
        if validation_dataset_adaptor is not None:
            # Add a metric calculator.
            self.val_dataset_adaptor = AgMLDatasetAdaptor(validation_dataset_adaptor)
            self.map = MAP()
        self._sanity_check_passed = False

    @auto_move_data
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # Run a forward pass through the model.
        images, annotations, _, _ = batch
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
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)
        detections = outputs["detections"]

        # Update the metric.
        if self.val_dataset_adaptor is not None and self._sanity_check_passed:
            for idx in image_ids:
                image, truth_boxes, truth_cls, _ = (
                    self.val_dataset_adaptor.get_image_and_labels_by_idx(idx)
                )
                pred_box, pred_labels, pred_conf = self.predict([image])
                if not isinstance(pred_labels[0], float):
                    pred_box, pred_labels, pred_conf = (
                        pred_box[0],
                        pred_labels[0],
                        pred_conf[0],
                    )
                if truth_cls.ndim == 0:
                    truth_cls = np.expand_dims(truth_cls, 0)
                metric_update_values = (
                    dict(
                        boxes=torch.tensor(pred_box, dtype=torch.float32),
                        labels=torch.tensor(pred_labels, dtype=torch.int32),
                        scores=torch.tensor(pred_conf),
                    ),
                    dict(
                        boxes=torch.tensor(truth_boxes, dtype=torch.float32),
                        labels=torch.tensor(truth_cls, dtype=torch.int32),
                    ),
                )
                self.map.update(*metric_update_values)

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
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

    def predict(self, images: Union[torch.Tensor, List]):
        """Runs inference on a set of images.

        Parameters
        ----------
        images : {torch.Tensor, list}
            Either a list of images (which can be numpy arrays, tensors, or
            another type), or a torch.Tensor returned from a DataLoader.

        Returns
        -------
        A tuple containing bounding boxes, confidence scores, and class labels.
        """
        if isinstance(images, list):
            image_sizes = [(image.size[1], image.size[0]) for image in images]
            images_tensor = torch.stack(
                [
                    self.inference_tfms(
                        image=np.array(image, dtype=np.float32),
                    )["image"]
                    for image in images
                ]
            )
            return self._run_inference(images_tensor, image_sizes)
        elif isinstance(images, torch.Tensor):
            image_tensor = images
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            if (
                image_tensor.shape[-1] != IMAGE_SIZE
                or image_tensor.shape[-2] != IMAGE_SIZE
            ):
                raise ValueError(
                    f"Input tensors must be of shape "
                    f"(N, 3, {IMAGE_SIZE}, {IMAGE_SIZE})"
                )

            num_images = image_tensor.shape[0]
            image_sizes = [(IMAGE_SIZE, IMAGE_SIZE)] * num_images
            return self._run_inference(image_tensor, image_sizes)
        else:
            raise TypeError(
                "Expected either a list of images or a "
                "torch.Tensor of images for `predict()`."
            )

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            images_tensor.shape[0], self.device, IMAGE_SIZE
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = (
            self.post_process_detections(detections)
        )
        scaled_bboxes = self._rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )
        return scaled_bboxes, predicted_class_labels, predicted_class_confidences

    @staticmethod
    def _create_dummy_inference_targets(num_images, device, size):
        return {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
                for _ in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=device) for _ in range(num_images)],
            "img_size": torch.tensor(
                [(size, size)] * num_images, device=device
            ).float(),
            "img_scale": torch.ones(num_images, device=device).float(),
        }

    def post_process_detections(self, detections):
        predictions = [
            self._postprocess_single_prediction_detections(d) for d in detections
        ]

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = (
            self.run_wbf(
                predictions, image_size=IMAGE_SIZE, iou_thr=self.wbf_iou_threshold
            )
        )
        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        # Extract the bounding boxes, confidence scores,
        # and class labels from the output detections.
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]

        # Only return boxes which are above the confidence threshold.
        valid_indexes = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[valid_indexes]
        scores = scores[valid_indexes]
        classes = classes[valid_indexes]
        return {"boxes": boxes, "scores": scores, "classes": classes}

    @staticmethod
    def _rescale_bboxes(predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims
            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / IMAGE_SIZE,
                            im_h / IMAGE_SIZE,
                            im_w / IMAGE_SIZE,
                            im_h / IMAGE_SIZE,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)
        return scaled_bboxes

    @staticmethod
    def run_wbf(
        predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None
    ):
        bboxes, confidences, class_labels = [], [], []

        for prediction in predictions:
            boxes = [(prediction["boxes"] / image_size).tolist()]
            scores = [prediction["scores"].tolist()]
            labels = [prediction["classes"].tolist()]

            boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
                boxes,
                scores,
                labels,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
            boxes = boxes * (image_size - 1)
            bboxes.append(boxes.tolist())
            confidences.append(scores.tolist())
            class_labels.append(labels.tolist())

        return bboxes, confidences, class_labels

    def on_validation_epoch_end(self) -> None:
        if not self._sanity_check_passed:
            self._sanity_check_passed = True
            return
        if hasattr(self, "metric_logger"):
            self.metric_logger.compile_epoch()
        if hasattr(self, "map"):
            map = self.map.compute().detach().cpu().numpy().item()
            self.log(
                "map", map, prog_bar=True, on_epoch=True, logger=True, sync_dist=True
            )
            self.map.reset()

    def on_fit_end(self) -> None:
        if hasattr(self, "metric_logger"):
            self.metric_logger.save()
        self.map.reset()

    def get_progress_bar_dict(self):
        p_bar = super(EfficientDetModel, self).get_progress_bar_dict()
        p_bar.pop("v_num", None)
        return p_bar


# Calculate and log the metrics.
class DetectionMetricLogger(MetricLogger):
    def update_metrics(self, y_pred, y_true) -> None:
        self.metrics["map"].update(y_pred, y_true)


def train(dataset, epochs, save_dir=None, overwrite=None, generalize_detections=False):
    """Constructs the training loop and trains a model."""
    save_dir = checkpoint_dir(None, save_dir)
    log_dir = save_dir.replace("checkpoints", "logs")

    # Check if the dataset already has benchmarks.
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        if not overwrite and len(os.listdir(save_dir)) >= 4:
            print(
                f"Checkpoints already exist for {dataset} "
                f"at {save_dir}, skipping generation."
            )
            return

    # Set up the checkpoint saving callback.
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            mode="min",
            filename=f"{dataset}" + "-epoch{epoch:02d}-valid_loss_{valid_loss:.2f}",
            monitor="valid_loss",
            save_top_k=3,
            auto_insert_metric_name=False,
        )
    ]

    # Create the loggers.
    loggers = [CSVLogger(log_dir), TensorBoardLogger(log_dir)]

    # Construct the data.
    print(f"Saving to {log_dir}.")
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(dataset)
    loader.shuffle()
    num_classes = loader.num_classes
    if generalize_detections:
        print("Generalizing class detections.")
        loader.generalize_class_detections()
        num_classes = 1
    loader.split(train=0.8, val=0.1, test=0.1)
    dm = EfficientDetDataModule(
        train_dataset_adaptor=AgMLDatasetAdaptor(loader.train_data),
        validation_dataset_adaptor=AgMLDatasetAdaptor(loader.val_data),
        num_workers=12,
        batch_size=4,
    )

    # Construct the model.
    model = EfficientDetModel(
        num_classes=num_classes,
        architecture="tf_efficientdet_d4",
        save_dir=save_dir,
        pretrained=True,
        validation_dataset_adaptor=loader.val_data,
    )

    # Create the trainer and train the model.
    msg = f"Training dataset {dataset}!"
    print("\n" + "=" * len(msg) + "\n" + msg + "\n" + "=" * len(msg) + "\n")
    trainer = pl.Trainer(
        max_epochs=epochs, gpus=gpus(None), callbacks=callbacks, logger=loggers
    )
    trainer.fit(model, dm)

    # Save the model state.
    torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pth"))


if __name__ == "__main__":
    # Parse input arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, nargs="+", help="The name of the dataset.")
    ap.add_argument(
        "--regenerate-existing",
        action="store_true",
        default=False,
        help="Whether to re-generate existing benchmarks.",
    )
    ap.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="The checkpoint directory to save to.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="How many epochs to train for. Default is 20.",
    )
    ap.add_argument(
        "--generalize-detections",
        action="store_true",
        default=False,
        help="Whether to generalize class labels.",
    )
    args = ap.parse_args()

    # Train the model.
    if args.dataset[0] == "except":
        exclude_datasets = args.dataset[1:]
        datasets = [
            dataset.name
            for dataset in agml.data.public_data_sources(ml_task="object_detection")
            if dataset.name not in exclude_datasets
        ]
    else:
        datasets = args.dataset
    train(
        dataset=datasets,
        epochs=args.epochs,
        save_dir=args.checkpoint_dir,
        overwrite=args.regenerate_existing,
        generalize_detections=args.generalize_detections,
    )
