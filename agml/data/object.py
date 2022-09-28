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
import abc
from functools import wraps

import cv2
import numpy as np

from agml.framework import AgMLSerializable
from agml.utils.image import imread_context
from agml.backend.tftorch import scalar_unpack


class DataObject(AgMLSerializable):
    """Stores a single piece of data and its corresponding annotation.

    This class stores an image and its corresponding annotation
    depending on the task, such as a label for a image classification
    task, a COCO JSON dictionary for an object detection task, or
    a mask for a semantic segmentation task.

    Fundamentally, it works as a two-object list, but the data is stored
    internally in a specific representation and images are only loaded
    lazily when necessary. This object is used internally in the
    `AgMLDataLoader`, and objects returned from it are returned as their
    expected contents (NumPy arrays/dictionaries/integers) when necessary.
    """
    serializable = frozenset(
        ('image_object', 'annotation_obj', 'dataset_root'))
    _abstract = frozenset(('_load_image_input', '_parse_annotation'))

    def __init__(self, image, annotation, root):
        # The `image` parameter is constant among different tasks.
        self._image_object = image

        # The `annotation` parameter varies with the task.
        self._annotation_obj = annotation

        # The `root` is the local root of the dataset. This is used
        # for object detection datasets primarily, whose COCO JSON
        # dictionary doesn't contain the full path, only the base.
        self._dataset_root = root

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return self.get()[i]

    def __repr__(self):
        return f"<DataObject: {self._image_object}, {self._annotation_obj}>"

    @staticmethod
    def _parse_image(path):
        with imread_context(os.path.abspath(path)) as image:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _parse_depth_image(path):
        with imread_context(os.path.abspath(path), flags = -1) as image:
            return image.astype(np.int32)

    @staticmethod
    def _parse_spectral_image(path):
        None # noqa, prevents `all abstract methods must be implemented`
        raise NotImplementedError("Multi/Hyperspectral images are not yet supported.")

    def get(self):
        """Returns the image and annotation pair with applied transforms.

        This method is the main exposed method to process the data. It loads
        the image and processes the annotation, then applies transformations.
        """
        return self._load()

    def _load(self):
        """Loads the image and annotation and returns them."""
        image = self._load_image_input(self._image_object)
        annotation = self._parse_annotation(self._annotation_obj)
        return image, annotation

    def __init_subclass__(cls, **kwargs):
        # Wraps derived abstract methods with the corresponding
        # docstring from this `DataObject` class and updates
        # the derived class' dictionary.
        wrapped_updates = {}
        self = super(DataObject, cls).__thisclass__ # noqa
        for name, method in cls.__dict__.items():
            if name in cls._abstract:
                wrapped = wraps(getattr(self, name))(method)
                wrapped_updates[name] = wrapped
        for name, method in wrapped_updates.items(): # noqa
            setattr(cls, name, method)

    @staticmethod
    def create(contents, task, root):
        """Creates a new `DataObject` for the corresponding task."""
        if task == 'image_classification':
            cls = ImageClassificationDataObject
        elif task == 'image_regression':
            cls = ImageRegressionDataObject
        elif task == 'object_detection':
            cls = ObjectDetectionDataObject
        elif task == 'semantic_segmentation':
            cls = SemanticSegmentationDataObject
        else:
            raise ValueError(f"Unsupported task {task}.")
        return cls(*contents, root)

    # The following methods are used to load the image and annotation.

    @abc.abstractmethod
    def _load_image_input(self, path):
        """Loads image inputs based on the task. Derived by subclasses."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _parse_annotation(self, obj):
        """Parses an annotation based on the task. Derived by subclasses."""
        raise NotImplementedError()

    # The following methods function independently whenever an
    # annotation is loaded. Depending on the task, the appropriate
    # method is called and a corresponding piece of data is returned.

    @staticmethod
    def _parse_label(obj):
        """Parses a label for an image classification task."""
        try:
            obj = int(obj)
        except TypeError:
            raise Exception(f"Could not convert object {obj} of "
                            f"type {type(obj)} to a scalar integer "
                            f"for image classification.")
        return obj

    @staticmethod
    def _parse_coco(obj):
        """Parses a COCO JSON dictionary for an object detection task."""
        annotation = {'bbox': [], 'category_id': [], 'area': [],
                      'image_id': "", 'iscrowd': [], 'segmentation': []}
        for a_set in obj:
            x, y, w, h = a_set['bbox']
            x = int(np.clip(x, 0, None))
            y = int(np.clip(y, 0, None))
            a_set['bbox'] = [x, y, w, h]
            annotation['bbox'].append(a_set['bbox'])
            annotation['category_id'].append(a_set['category_id'])
            annotation['iscrowd'].append(a_set['iscrowd'])
            annotation['segmentation'].append(a_set['segmentation'])
            annotation['area'].append(a_set['area'])
            annotation['image_id'] = a_set['image_id']
        for key, value in annotation.items():
            # Creating nested sequences from ragged arrays (see numpy).
            if key in ['segmentation']:
                out = np.array(value, dtype = object)
            else:
                out = np.array(value)
                if np.isscalar(out):
                    out = scalar_unpack(out)
            annotation[key] = out
        return annotation

    @staticmethod
    def _parse_mask(obj):
        """Parses a mask for a semantic segmentation task."""
        with imread_context(os.path.realpath(obj)) as image:
            if image.ndim == 3:
                if not np.all(image[:, :, 0] == image[:, :, 1]):
                    raise TypeError(
                        f"Invalid annotation mask of shape {image.shape}.")
                image = image[:, :, 0]
            return np.squeeze(image)


class ImageClassificationDataObject(DataObject):
    """Serves as a `DataObject` for image classification tasks."""
    def _load_image_input(self, path):
        return self._parse_image(path)

    def _parse_annotation(self, obj):
        return self._parse_label(obj)


class ImageRegressionDataObject(DataObject):
    """Serves as a `DataObject` for image regression tasks."""
    def _load_image_input(self, contents):
        # The easy case, when there is only one input image.
        if isinstance(contents, str) and os.path.exists(contents):
            return self._parse_image(contents)

        # Otherwise, we have a dictionary containing multiple
        # input types, so we need to independently load those.
        images = dict.fromkeys(contents.keys(), None)
        for c_type, path in contents.items():
            if c_type == 'image':
                images[c_type] = self._parse_image(path)
            elif c_type == 'depth_image':
                images[c_type] = self._parse_depth_image(path)
            else:
                images[c_type] = self._parse_spectral_image(path)
        return images

    def _parse_annotation(self, obj):
        return obj


class ObjectDetectionDataObject(DataObject):
    """Serves as a `DataObject` from object detection tasks."""
    def _load_image_input(self, path):
        path = os.path.join(self._dataset_root, 'images', path)
        return self._parse_image(path)

    def _parse_annotation(self, obj):
        return self._parse_coco(obj)


class SemanticSegmentationDataObject(DataObject):
    """Serves as a `DataObject` for semantic segmentation tasks."""
    def _load_image_input(self, path):
        return self._parse_image(path)

    def _parse_annotation(self, obj):
        return self._parse_mask(obj)



