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

from typing import Any, List, Union

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

import agml


def _pre_prepare_for_efficientdet(image, annotation):
    """Prepares the image and annotation for EfficientDet.

    This preparation stage occurs *pre-transformation*.
    """
    # Convert the image type.
    image = image.astype(np.uint8)

    # Clip the bounding boxes to the image shape to prevent errors.
    bboxes = np.array(annotation["bbox"]).astype(np.int32)
    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    x_max = bboxes[:, 2] + x_min
    y_max = bboxes[:, 3] + y_min
    x_min, y_min = np.clip(x_min, 0, image.shape[1]), np.clip(y_min, 0, image.shape[0])
    x_max, y_max = np.clip(x_max, 0, image.shape[1]), np.clip(y_max, 0, image.shape[0])

    # Reconstruct the boxes and get the class labels.
    bboxes = np.dstack((x_min, y_min, x_max, y_max)).squeeze(axis=0)
    class_labels = np.array(annotation["category_id"]).squeeze()

    # Add an extra dimension to labels for consistency.
    if class_labels.ndim == 0:
        class_labels = np.expand_dims(class_labels, axis=0)

    # Construct and return the sample.
    return image, {"bboxes": bboxes, "labels": class_labels}


def _post_prepare_for_efficientdet(image, annotation):
    """Prepares the image and annotation for EfficientDet.

    This preparation stage occurs *post-transformation*.
    """
    bboxes = np.array(annotation["bboxes"])
    labels = annotation["labels"]

    # Convert 1-channel and 4-channel to 3-channel.
    if image.shape[0] == 1:
        image = torch.tile(image, (3, 1, 1))
    if image.shape[0] == 4:
        image = image[:3]

    # Convert to yxyx from xyxy.
    _, new_h, new_w = image.shape
    bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

    # Create the target from the annotations.
    target = {
        "bboxes": torch.as_tensor(bboxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels),
        "img_size": torch.tensor([new_h, new_w]),
        "img_scale": torch.tensor([1.0]),
    }
    return image, target


class TransformApplier(object):
    """Applies transforms to the data."""

    def __init__(self, augmentations: Any, image_size: int = 512):
        self._image_size = image_size
        if augmentations is None:
            augmentations = self._default_augmentation
        self._augmentations = augmentations

    def _default_augmentation(self, image, bboxes, labels):
        # Construct the sample.
        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": bboxes,
            "labels": labels,
        }

        # Augment the sample.
        sample = A.Compose(
            [
                A.Resize(height=self._image_size, width=self._image_size, p=1),
                ToTensorV2(p=1),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0,
                min_visibility=0,
                label_fields=["labels"],
            ),
        )(**sample)

        # Return the sample.
        return sample["image"], {"bboxes": sample["bboxes"], "labels": sample["labels"]}

    @staticmethod
    def _unpack_result(result):
        if isinstance(result, dict):
            return result["image"], {
                "bboxes": result["bboxes"],
                "labels": result["labels"],
            }
        return result

    def _apply_train(self, image, annotation):
        image, annotation = _pre_prepare_for_efficientdet(image, annotation)
        image, annotation = self._unpack_result(
            self._augmentations["train"](
                image=image, bboxes=annotation["bboxes"], labels=annotation["labels"]
            )
        )
        image, annotation = _post_prepare_for_efficientdet(image, annotation)
        return image, annotation

    def _apply_val(self, image, annotation):
        image, annotation = _pre_prepare_for_efficientdet(image, annotation)
        image, annotation = self._unpack_result(
            self._augmentations["val"](
                image=image, bboxes=annotation["bboxes"], labels=annotation["labels"]
            )
        )
        image, annotation = _post_prepare_for_efficientdet(image, annotation)
        return image, annotation

    def _apply(self, image, annotation):
        image, annotation = _pre_prepare_for_efficientdet(image, annotation)
        image, annotation = self._unpack_result(
            self._augmentations(
                image=image, bboxes=annotation["bboxes"], labels=annotation["labels"]
            )
        )  # noqa
        image, annotation = _post_prepare_for_efficientdet(image, annotation)
        return image, annotation


def build_loader(
    dataset: Union[List[str], str], batch_size: int = 4
) -> agml.data.AgMLDataLoader:
    """Constructs an `AgMLDataLoader` for object detection.

    Given either a single dataset or a list of datasets, this method will
    construct an `AgMLDataLoader` which is prepared for object detection
    tasks. This includes image and annotation formatting.
    """
    # Construct the loader from the input dataset.
    pl.seed_everything(2499751)
    loader = agml.data.AgMLDataLoader(dataset)
    loader.shuffle()

    # Apply the batch size.
    loader.batch(batch_size=batch_size)

    # Split and return the loader.
    loader.split(train=0.8, val=0.1, test=0.1)
    return loader


class EfficientDetDataModule(pl.LightningDataModule):
    """Wraps an `AgMLDataLoader` into a `LightningDataModule`."""

    def __new__(cls, *args, **kwargs):
        if kwargs.get("no_agml", False):
            return EfficientDetDataModuleNoAgML(**kwargs)
        return super(EfficientDetDataModule, cls).__new__(cls, *args, **kwargs)

    def __init__(
        self,
        loader: agml.data.AgMLDataLoader,
        augmentation: Any = None,
        num_workers: int = 8,
        **kwargs,
    ):
        # Get the training and validation `AgMLDataLoader`s.
        self._train_loader = loader.train_data
        self._train_loader.as_torch_dataset()
        self._val_loader = loader.val_data
        self._val_loader.as_torch_dataset()

        # Update the transforms.
        if isinstance(augmentation._augmentations, dict):
            self._train_transform = augmentation._apply_train
            self._val_transform = augmentation._apply_val
        else:
            self._train_transform = self._val_transform = augmentation._apply
        self._train_loader.transform(dual_transform=self._train_transform)
        self._val_loader.transform(dual_transform=self._val_transform)

        # Initialize the base module.
        self._num_workers = num_workers
        super(EfficientDetDataModule, self).__init__()

    @property
    def train_loader(self):
        return self._train_loader

    def train_dataset(self) -> Dataset:
        return self._train_loader

    def train_dataloader(self) -> DataLoader:
        return self.train_dataset().export_torch(
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self._num_workers,
            collate_fn=self.collate_fn,
        )

    @property
    def val_loader(self):
        return self._val_loader

    def val_dataset(self) -> Dataset:
        return self._val_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_dataset().export_torch(
            pin_memory=True,
            drop_last=True,
            num_workers=self._num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """Collates items together into a batch."""
        images, targets = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.stack([target["img_size"] for target in targets]).float()
        img_scale = torch.stack([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }
        return images, annotations, targets


class EfficientDetDataModuleNoAgML(pl.LightningDataModule):
    """Wraps a non-AgMLDataLoader dataset."""

    def __init__(self, **kwargs):
        # A custom loader is passed.
        self._train_ds = kwargs["train_loader"]
        self._val_ds = kwargs["val_loader"]
        self._num_workers = kwargs["num_workers"]
        super(EfficientDetDataModuleNoAgML, self).__init__()

    def train_dataset(self) -> Dataset:
        return self._train_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(),
            batch_size=2,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self._num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataset(self) -> Dataset:
        return self._val_ds

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset(),
            batch_size=2,
            pin_memory=True,
            drop_last=True,
            num_workers=self._num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """Collates items together into a batch."""
        images, targets = tuple(zip(*batch))
        images = [
            torch.tensor(image) if not isinstance(image, torch.Tensor) else image
            for image in images
        ]
        images = torch.stack([image for image in images])
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.stack([target["img_size"] for target in targets]).float()
        img_scale = torch.stack([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }
        return images, annotations, targets
