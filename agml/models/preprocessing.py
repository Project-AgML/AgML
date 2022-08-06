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
A set of data preprocessing functions for `AgMLDataLoaders`.
"""

import inspect

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EfficientDetPreprocessor(object):
    """A preprocessor which prepares a data sample for `EfficientDet`.
    
    This class can be used to construct a preprocessing pipeline which
    auto-formats the data in an `AgMLDataLoader` for object detection into
    the format necessary for training an `EfficientDet` model. By default,
    this includes resizing images, converting bounding boxes to `yxyx`, and
    finally preparing the image and annotation for PyTorch.
    
    Using this class supersedes the need for any other transformations or
    even image resizing. It can be used as follows:
    
    > loader = agml.data.AgMLDataLoader('grape_detection_californiaday')
    > processor = agml.models.EfficientDetPreprocessor(
    >     image_size = 512, augmentation = [A.HorizontalFlip(0.5)])
    > loader.transform(dual_transform = processor)
    
    Parameters
    ----------
    image_size : int
        The size to which images will be resized (default of 512).
    augmentation : Any
        Either a list of albumentations transforms (without being wrapped
        into a Compose object), or a custom method which accepts three
        arguments: `image`, `bboxes`, and `labels`, and returns the same.
        The `bboxes` will be in the XYXY format.
    
    Notes
    -----
    - Passing `augmentation = None`, the default, is equivalent to preparing
      a validation or test loader: preprocessing is applied, but no transforms.
    - Note that if you pass a custom augmentation method, the resulting output
      is expected to be in PyTorch's format (image should be a tensor with its
      first dimension being the image's channels, for example).
    """

    def __init__(self, image_size = 512, augmentation = None):
        # Parse the image size.
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)):
            if not len(image_size) == 2:
                raise ValueError(
                    "Expected either an integer or sequence of 2 values "
                    "for `image_size`, instead got ({image_size}).")

        # Construct the applied input augmentation.
        self._albumentations_params = dict(
            bbox_params = A.BboxParams(
                format = "pascal_voc", min_area = 0,
                min_visibility = 0, label_fields = ["labels"]),
            standard_augmentations = [
                A.Resize(height = image_size[0],
                         width = image_size[1], p = 1),
                ToTensorV2(p = 1)])
        self._check_and_make_augmentation(augmentation)

    def _check_and_make_augmentation(self, augmentation):
        """Constructs the applied augmentation from the inputs."""
        # If no augmentation is provided, then use the defaults.
        if augmentation is None:
            self._augmentation = A.Compose([
                *self._albumentations_params['standard_augmentations']
            ], p = 1.0, bbox_params = self._albumentations_params['bbox_params'])

        # If a list of augmentations are provided, then use those augmentations
        # wrapped alongside the default ones in a `Compose` object.
        elif isinstance(augmentation, list):
            if not all(isinstance(a, A.BasicTransform) for a in augmentation):
                raise ValueError(
                    "If providing a list of transforms, all of them must be "
                    f"albumentations augmentations, instead got {augmentation} "
                    f"of types {[type(i) for i in augmentation]}.")
            self._augmentation = A.Compose([
                *augmentation, *self._albumentations_params['standard_augmentations']
            ], p = 1.0, bbox_params = self._albumentations_params['bbox_params'])

        # Otherwise, the augmentation should be a method with three input
        # arguments, so check it and then wrap it into an application method.
        else:
            if not len(inspect.signature(augmentation).parameters):
                raise ValueError(
                    f"The input augmentation should have three input arguments, "
                    f"instead got {inspect.signature(augmentation).parameters}.")
            self._method_augmentation = augmentation
            self._augmentation = self._apply_method_augmentation

    def _apply_method_augmentation(self, image, bboxes, labels):
        # Wrapper method to apply a user-provided method augmentation.
        image, bboxes, labels = self._method_augmentation(image, bboxes, labels)
        return {'image': image, 'bboxes': bboxes, 'labels': labels}

    def __call__(self, image, annotation):
        # Convert the image type.
        image = Image.fromarray(image)

        # Clip the bounding boxes to the image shape to prevent errors.
        bboxes = np.array(annotation['bbox']).astype(np.int32)
        x_min = bboxes[:, 0]
        y_min = bboxes[:, 1]
        x_max = bboxes[:, 2] + x_min
        y_max = bboxes[:, 3] + y_min
        x_min, y_min = np.clip(x_min, 0, image.width), \
                       np.clip(y_min, 0, image.height)
        x_max, y_max = np.clip(x_max, 0, image.width), \
                       np.clip(y_max, 0, image.height)

        # Reconstruct the boxes and get the class labels.
        bboxes = np.dstack((x_min, y_min, x_max, y_max)).squeeze(axis = 0)
        class_labels = np.array(annotation['category_id']).squeeze()

        # Add an extra dimension to labels for consistency.
        if class_labels.ndim == 0:
            class_labels = np.expand_dims(class_labels, axis = 0)

        # Add an albumentations augmentation.
        sample = {'image': np.array(image, dtype = np.float32),
                  'bboxes': bboxes, 'labels': class_labels}
        sample = self._augmentation(**sample)
        image = sample['image']
        bboxes = np.array(sample['bboxes'])
        labels = sample['labels']

        # Convert 1-channel and 4-channel to 3-channel.
        if image.shape[0] == 1:
            image = torch.tile(image, (3, 1, 1))
        if image.shape[0] == 4:
            image = image[:3]

        # Convert to yxyx from xyxy.
        _, new_h, new_w = image.shape
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis = 0)
        bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

        # Create the target from the annotations.
        target = {
            "bboxes": torch.as_tensor(
                bboxes, dtype = torch.float32),
            "labels": torch.as_tensor(labels),
            "img_size": torch.tensor([new_h, new_w]),
            "img_scale": torch.tensor([1.0])}
        return image, target


