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
import pickle

import cv2
import numpy as np

from agml.framework import AgMLSerializable
from agml.utils.logging import log
from agml.utils.image import imread_context
from agml.utils.general import resolve_tuple
from agml.utils.io import recursive_dirname


class ImageResizeManager(AgMLSerializable):
    """Smartly resizes input image and annotation data.

    This class wraps the resizing of image and annotation data and
    dynamically resizes images and annotations based on the user
    preference. Resizing is applies to input images and annotations
    where required, e.g. masks for semantic segmentation, or COCO
    JSON bounding boxes for object detection.

    Image resizing contains a few modes:

    1. `default`: No resizing, leaves the images in their default size.
       This is the default parameter if nothing is passed.
    2. `train`: This will set a default training size of (512, 512).
    3. `imagenet`: This will set a default size of (224, 224).
    4. custom size: Resizes the images to the provided size.
    5. `auto`: Dynamically selects an image size based on a few factors.
       For example, if there are certain outliers in a dataset which are
       of a different size while the majority remain the same, then, the
       behavior of this method is chosen by a majority threshold of the
       sizes of all the images in the dataset. If no shape can be inferenced,
       it returns a default size of (512, 512). The logic for the `auto`
       case is explained in more detail below.

    The actual resizing caveats are as follows:

    - For object detection, the bounding box coordinates will
      be resized and the area of the box will in turn be recomputed.
    - For semantic segmentation, the annotation mask will be resized,
      using a nearest-neighbor interpolation to keep it as similar
      as possible to the original mask (preventing data loss).

    Note: Images will be resized *before* being passed into
    a transformation pipeline. This is to prevent data loss.
    """
    serializable = frozenset(
        ('task', 'dataset_name', 'dataset_root', 'auto_enabled',
         'resize_type', 'image_size', 'interpolation'))

    # Stores the path to the local file which contains the
    # information on the image shapes in all of the datasets.
    _shape_info_file = os.path.join(
        recursive_dirname(__file__, 3),
        '_assets', 'shape_info.pickle')

    # Stores the default image size if it is not inferenced.
    _default_size = (512, 512)

    # Stores the default imagenet model input shape.
    _imagenet_size = (224, 224)

    def __init__(self, task, dataset, root):
        self._task = task
        self._dataset_name = dataset
        self._dataset_root = root

        self._resize_type = 'default'
        self._image_size = None
        self._interpolation = cv2.INTER_LINEAR

        self._auto_enabled = True

    def disable_auto(self):
        # For multi-dataset loaders, the `auto` option is disabled.
        self._auto_enabled = False

    @property
    def state(self):
        return self._resize_type

    @property
    def size(self):
        return self._image_size

    @staticmethod
    def _tuple_euclidean(t1, t2):
        (x1, y1), (x2, y2) = t1, t2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _method_resize(self, image, size):
        return cv2.resize(image, size, interpolation = self._interpolation)

    def assign(self, kind, method = None):
        """Assigns the resize parameter (and does necessary calculations)."""
        if kind == 'default':
            self._resize_type = 'default'
            self._image_size = None
        elif kind == 'train':
            self._resize_type = 'train'
            self._image_size = self._default_size
        elif kind == 'imagenet':
            self._resize_type = 'imagenet'
            self._image_size = self._imagenet_size
        elif isinstance(kind, (list, tuple, np.ndarray, set)):
            if not len(kind) == 2:
                raise ValueError(
                    f"Got a sequence {kind}, expected two values for "
                    f"the height and width but got {len(kind)} values.")
            self._resize_type = 'custom_size'
            self._image_size = tuple([i for i in kind])
        elif 'auto' in kind:
            # This means that the dataloader has been exported in some format.
            # First, we check whether a different size has automatically been
            # set, since if it has, we don't want to override that size.
            if kind == 'train-auto':
                if not self._auto_enabled:
                    self._resize_type = 'train'
                    self._image_size = self._default_size
                elif self._resize_type == 'default':
                    info = self._maybe_load_shape_info()
                    if info is None:
                        shape = self._random_inference_shape()
                    else:
                        shape = self._inference_shape(info)
                    self._resize_type = 'auto'
                    self._image_size = resolve_tuple(shape)

        # Check interpolation method independently.
        if method is not None:
            method = method.lower()
            if method not in ['bilinear', 'area', 'nearest', 'cubic']:
                raise ValueError(f"Invalid interpolation method: {method}.")
            self._interpolation = {
                'bilinear': cv2.INTER_LINEAR,
                'area': cv2.INTER_AREA,
                'nearest': cv2.INTER_NEAREST,
                'cubic': cv2.INTER_CUBIC
            }[method]

    def apply(self, contents):
        """Applies the resizing operation to the input data."""
        if self._task in ['image_classification', 'image_regression']:
            return self._resize_image_input(
                contents, self._image_size)
        elif self._task == 'object_detection':
            return self._resize_image_and_coco(
                contents, self._image_size)
        elif self._task == 'semantic_segmentation':
            return self._resize_image_and_mask(
                contents, self._image_size)

    def _inference_shape(self, info):
        """Attempts to inference a shape for the `auto` method.

        The logic for this method goes in the following order:

        1. If there is only one unique shape in the dataset, then that
           is the shape that is used.
        2. If one shape makes up more than 70% of the shapes in the
           dataset, (7/10 of the dataset), then that shape is used.
        3. If there are less than or equal to 4 shapes which make up
           at least 10% of the dataset each (e.g., there may be 40%
           one shape, 40% another shape, and then 20% a third and final
           shape), then the shape closest to (512, 512) will be used. If
           there is one that makes up a majority, however, so greater
           than 50% of the dataset, and the next-largest is less than 30%,
           then that majority size will be returned.

        4. Otherwise, no specific shape can be inferenced, so the default
           shape of (512, 512) is used instead.

        Note that the `auto` method is very experimental, and is more useful
        just for instantly determining when a dataset has only one type
        of shape rather than having to manually figure sizes out. If you
        encounter issues in training when there are multiple pluralities
        of shapes and no explicitly majority, then do not use this method.
        """
        shapes, counts = info
        shapes = shapes[:, :-1]
        if len(shapes) == 1:
            return shapes[0].tolist()

        total_count = np.sum(counts)
        if np.max(counts) > int(total_count * 0.7):
            return shapes[np.where(counts == np.max(counts))[0]][0].tolist()

        if len(shapes) <= 4:
            count_proportions = counts.copy() / total_count
            maybe_max = np.where(count_proportions >= 0.5)[0]
            if maybe_max:
                if maybe_max.copy().sort()[-2] <= 0.3:
                    return shapes[maybe_max].tolist()

            two_pluralities = shapes[np.where(count_proportions >= 0.35)[0]]
            if len(two_pluralities) != 2:
                return self._get_log_default_shape()
            outs = (self._tuple_euclidean(two_pluralities[0], self._default_size),
                    self._tuple_euclidean(two_pluralities[1], self._default_size))
            return two_pluralities[outs.index(max(outs))][0].tolist()

        return self._get_log_default_shape()

    def _random_inference_shape(self):
        """Randomly attempts to inference a shape for the dataset.

        In the case that there is no shape information saved in the
        local `.shape.info.pickle` file, this method randomly attempts
        to inference a dataset shape by randomly selecting images from
        the dataset and performing calculations (similarly to what it
        would do in the regular shape inference method, only with a
        random sample instead of the entire dataset for reference.
        """
        log(f"Could not find shape information for {self._dataset_name}. "
            f"Attempting to randomly inference the dataset shape.")

        image_path = os.path.join(self._dataset_root, 'images')
        images = np.random.choice(os.listdir(image_path), size = 25)

        # Get all of the shapes from the random sample of images.
        shapes = []
        for path in images:
            path = os.path.join(image_path, path)
            with imread_context(path) as image:
                shapes.append(image)

        # Inference a valid shape from the shapes. We dispatch to the
        # regular inferencing method once we have the shapes and counts.
        unique_shapes, counts = np.unique(shapes, return_counts = True, axis = 0)
        return self._inference_shape((unique_shapes, counts))

    def _maybe_load_shape_info(self):
        """Loads the contents of the shape information file."""
        try:
            with open(self._shape_info_file, 'rb') as f:
                contents = pickle.load(f)
                return contents.get(self._dataset_name, None)
        except OSError:
            raise EnvironmentError(
                f"Could not find the local file {self._shape_info_file} "
                f"containing shape information. There was likely a problem "
                f"when building AgML. Please re-install it.")

    def _get_log_default_shape(self):
        """Returns the default shape and warns that no shape could be inferenced."""
        log(f"No specific shape could be inferenced for the dataset "
            f"{self._dataset_name}. Using the default shape of (512, "
            f"512). Please use this instead of 'auto'. You can set this "
            f"shape by default by using the parameter 'train'.")
        return self._default_size

    # The following methods conduct the actual resizing of the images
    # and potentially their annotations for the different tasks.

    def _resize_image_input(self, contents, image_size):
        # If there is only one image input, just return that.
        image, label = contents
        if isinstance(image, np.ndarray):
            return self._resize_single_image(contents, image_size)
        if image_size is not None:
            return {
                k: self._method_resize(
                    i.astype(np.uint16), image_size).astype(np.int32)
                for k, i in image.items()}, label
        return image, label

    def _resize_single_image(self, contents, image_size):
        image, label = contents
        if image_size is not None:
            # No processing done on the annotation.
            image = self._method_resize(image, image_size)
        return image, label

    def _resize_image_and_coco(self, contents, image_size):
        image, coco = contents
        if image_size is not None:
            # Extract the original size of the bounding boxes.
            y_scale, x_scale = image.shape[0:2]
            bboxes = coco['bbox']
            processed_bboxes = []
            for bbox in bboxes:
                x1, y1, width, height = bbox
                processed_bboxes.append([
                    x1 / x_scale, y1 / y_scale,
                    width / x_scale, height / y_scale])

            # Clip the bounding boxes (There might be a case where
            # the bounding box is on the edge, but goes to just over 1.0
            # which in turn causes many bugs. This prevents that).
            processed_bboxes = np.array(processed_bboxes)
            np.clip(processed_bboxes, 0, 1, processed_bboxes)

            # Resize the image and calculate its new size ratio.
            image = self._method_resize(image, image_size)
            y_new, x_new = image.shape[0:2]

            # Update the bounding boxes and areas with the new ratio.
            final_processed_bboxes, areas = [], []
            for bbox in processed_bboxes:
                x1, y1, width, height = bbox
                new_bbox = [
                    int(x1 * x_new), int(y1 * y_new),
                    int(width * x_new), int(height * y_new)]
                final_processed_bboxes.append(new_bbox)
                areas.append(new_bbox[2] * new_bbox[3])
            final_processed_bboxes = \
                np.array(final_processed_bboxes).astype(np.int32)
            coco['bbox'] = final_processed_bboxes
            areas = np.array(areas).astype(np.int32)
            coco['area'] = areas
        return image, coco

    def _resize_image_and_mask(self, contents, image_size):
        image, mask = contents
        if image_size is not None:
            # Resize the image and the mask together if requested.
            image = self._method_resize(image, image_size)
            mask = cv2.resize(mask, image_size, cv2.INTER_NEAREST)
        return image, mask






