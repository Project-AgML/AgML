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

import abc

import numpy as np

from agml.framework import AgMLSerializable
from agml.utils.random import seed_context


class TransformApplierBase(AgMLSerializable):
    """Applies a transform to the input data.

    This is used as a wrapper class for applying transformations in
    more complex cases. Derived classes implement the `apply` method
    to wrap the input transformation.

    This is used in cases such as `albumentations` transforms, which
    require keyword arguments. This wrapper allows them to be used
    as a traditional method for consistency in application.
    """
    serializable = frozenset(('transform', ))

    def __init__(self, transform):
        self._transform = transform

    @abc.abstractmethod
    def apply(self, *args, **kwargs):
        """Applies the transformation to the input data."""
        return

    def __call__(self, *args):
        return self.apply(*args)

    def __str__(self):
        return self.__class__.__name__ + f": {self._transform}"


class AlbumentationsTransformSingle(TransformApplierBase):
    def apply(self, image):
        transform = self._transform(image = image)
        return transform['image']


class AlbumentationsTransformMask(TransformApplierBase):
    def apply(self, image, mask):
        transform = self._transform(image = image, mask = mask)
        return transform['image'], transform['mask']


class AlbumentationsTransformCOCO(TransformApplierBase):
    def apply(self, image, coco):
        # This method is a bit more complex. We can't just apply it
        # to the COCO JSON dictionary as we need to extract the
        # bounding boxes/category IDs, do the transformation on those,
        # and then re-insert them into the COCO dictionary.
        coco_boxes = coco['bbox']
        coco_labels = coco['category_id']
        bboxes = np.c_[coco_boxes, coco_labels]
        transform = self._transform(image = image, bboxes = bboxes)
        image, bboxes = transform['image'], \
                        np.array(transform['bboxes'])[:, :-1]
        coco['bbox'], areas = bboxes, []
        for box in bboxes:
            areas.append(
                (box[0] + box[2]) * (box[1] + box[3]))
        coco['area'] = areas.copy()
        return image, coco


class SameStateImageMaskTransform(TransformApplierBase):
    def apply(self, image, mask):
        # This method applies a transformation to the image and mask.
        # Essentially, we need to set the random seed before applying
        # it to the image and mask to get the same results for both
        # the image and the mask. So, we change the seed each time, but
        # we also reset the seed so that it goes back to normal afterwards.
        seed = np.random.randint((1 << 31) - 1) # default is 32-bit systems
        with seed_context(seed) as context:
            image = self._transform(image)
            context.reset()
            mask = self._transform(mask)
        return image, mask


class NormalizationTransformBase(TransformApplierBase, abc.ABC):
    """A subclass to mark transforms as normalizing transforms."""
    pass


class ScaleTransform(NormalizationTransformBase):
    def apply(self, image):
        if image.max() >= 1 or np.issubdtype(image.dtype, np.integer):
            image = (image / 255).astype(np.float32)
        return image


class NormalizationTransform(NormalizationTransformBase):
    def apply(self, image):
        # This method applies normalization to input images, scaling them
        # to a 0-1 float range, and performing scaling normalization.
        if image.max() >= 1 or np.issubdtype(image.dtype, np.integer):
            image = (image / 255).astype(np.float32)

        mean, std = self._transform
        mean = np.array(mean, dtype = np.float32)
        std = np.array(std, dtype = np.float32)
        denominator = np.reciprocal(std, dtype = np.float32)
        image = (image - mean) * denominator
        return image


class OneHotLabelTransform(TransformApplierBase):
    def apply(self, labels):
        # This applies a one-hot label transformation. The only argument
        # in the `_transform` parameter is simply the number of labels.
        one_hot = np.zeros(shape = (self._transform, ))
        one_hot[labels] = 1
        return one_hot.astype(np.float32)


class MaskToChannelBasisTransform(TransformApplierBase):
    def apply(self, mask):
        # Converts a 2-d mask of `n` labels, excluding background, to a
        # `n x h x w` 3-dimensional mask with each channel a label.
        input_shape = mask.shape
        mask = mask.ravel()
        n = mask.shape[0]
        mask = np.array(mask, dtype = np.int32)
        out = np.zeros(shape = (n, self._transform + 1))
        out[np.arange(n), mask] = 1
        out = np.reshape(out, input_shape + (self._transform + 1,))
        return out[..., 1:].astype(np.int32)


