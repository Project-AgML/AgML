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

import types
import inspect
import functools
from enum import Enum

from agml.framework import AgMLSerializable
from agml.backend.tftorch import (
    get_backend, set_backend, user_changed_backend, StrictBackendError
)
from agml.data.managers.transform_helpers import (
    AlbumentationsTransformSingle,
    AlbumentationsTransformMask,
    AlbumentationsTransformCOCO,
    SameStateImageMaskTransform,
    NormalizationTransformBase,
    ScaleTransform,
    NormalizationTransform,
    OneHotLabelTransform,
    MaskToChannelBasisTransform
)
from agml.utils.logging import log


class TransformKind(Enum):
    Transform = 'transform'
    TargetTransform = 'target_transform'
    DualTransform = 'dual_transform'


# Shorthand form of the enum for testing explicit values.
t_ = TransformKind


class TransformManager(AgMLSerializable):
    """Manages the validation and application of transformations.

    This class serves as a helper for the `DataManager` class that
    focuses solely on applying transforms to the input images and
    annotations.

    There are three main transforms stored internally.

    1. `transform`: A transform which is applied to only the input image.
    2. `target_transform`: A transform which is applied only to the output.
    3. `dual_transform`: A transform which is applied to both.

    Note that the values of these transforms don't always directly match
    the provided value to the corresponding argument in the method
    `AgMLDataLoader.transform()`. An example is an albumentations
    transform, which, while being passed to `transform`, ends up being
    stored as a `dual_transform` internally. The above case applies to
    semantic segmentation and object detection transformations, but
    there is no `dual_transform` for image classification.
    """
    serializable = frozenset(
        ('task', 'transforms', 'time_inserted_transforms'))

    def __init__(self, task):
        self._task = task
        self._transforms = dict()

        # A user might want to apply a certain set of transforms first,
        # which are then followed by a different set of transforms. E.g.,
        # calling `loader.transform()` with one set of transforms and then
        # following with another call to `loader.transform()` with the
        # intention that the transforms in the second call will only be
        # applied after all of the transforms in the first call are.
        #
        # So, while the transform types are tracked in the `_transforms`
        # attribute, we track the moment that transforms are inserted
        # using this attribute, which is a list of different transforms.
        # The `apply()` method loops sequentially through each transform
        # in this list and applies them as required.
        self._time_inserted_transforms = []

    def get_transform_states(self):
        """Returns a copy of the existing transforms."""
        transform_dict = {}
        for name, state in self._transforms.items():
            transform_dict[name] = state.copy()
        return transform_dict

    def _pop_transform(self, t_type, search_param):
        """Removes a certain type of transform from the manager."""
        for i, tfm in enumerate(self._transforms[search_param]):
            if isinstance(tfm, t_type):
                self._transforms[search_param].pop(i)
                break
        for i, tfm in enumerate(self._time_inserted_transforms):
            if isinstance(tfm[1], t_type):
                self._time_inserted_transforms.pop(i)
                break

    def assign(self, kind, transform):
        """Assigns a new transform to the manager."""
        # Determine if the transform is being reset or unchanged.
        if transform == 'reset':
            self._transforms.pop(kind, None)
            new_time_transforms = []
            for tfm in self._time_inserted_transforms:
                if tfm[0] == kind:
                    continue
                new_time_transforms.append(tfm)
            self._time_inserted_transforms = new_time_transforms.copy()
            return
        elif transform is None:
            return

        # If an `albumentations` transform is passed to the `transform`
        # argument, then it is checked first to see if it is potentially
        # just processing the input image, and then passed as just a 
        # `transform` (or else it will clash in object detection tasks).
        # Otherwise, it is stored internally as a `dual_transform`.
        if transform is not None:
            try:
                if 'albumentations' in transform.__module__:
                    if t_(kind) == TransformKind.Transform:
                        if len(transform.processors) != 0:
                            kind = 'dual_transform'
                    if self._task == 'semantic_segmentation':
                        kind = 'dual_transform'
            except AttributeError:
                # Some type of object that doesn't have `__module__`.
                pass

        # We can only do this after the albumentations check, to ensure
        # that we are adding the transforms to the correct location.
        prev = self._transforms.get(kind, None)

        # Validate the transformation based on the task and kind.
        if self._task == 'image_classification':
            if t_(kind) == TransformKind.Transform:
                transform = self._maybe_normalization_or_regular_transform(transform)
            elif t_(kind) == TransformKind.TargetTransform:
                if isinstance(transform, tuple): # a special convenience case
                    if transform[0] == 'one_hot':
                        if transform[2] is not True: # removing the transform
                            self._pop_transform(OneHotLabelTransform, kind)
                            return
                        transform = OneHotLabelTransform(transform[1])
            else:
                raise ValueError("There is no `dual_transform` for image "
                                 "classification tasks. Please pass the "
                                 "input as a `transform` or `target_transform`.")
        elif self._task == 'image_regression':
            if t_(kind) == TransformKind.Transform:
                transform = self._maybe_normalization_or_regular_transform(transform)
            elif t_(kind) == TransformKind.TargetTransform:
                if isinstance(transform, tuple): # a special convenience case
                    if transform[0] == 'one_hot':
                        if transform[2] is not True: # removing the transform
                            self._pop_transform(OneHotLabelTransform, kind)
                            return
                        transform = OneHotLabelTransform(transform[1])
            else:
                pass
        elif self._task == 'semantic_segmentation':
            if t_(kind) == TransformKind.Transform:
                transform = self._maybe_normalization_or_regular_transform(transform)
            elif t_(kind) == TransformKind.TargetTransform:
                if isinstance(transform, tuple): # a special convenience case
                    if transform[0] == 'channel_basis':
                        if transform[2] is not True: # removing the transform
                            self._pop_transform(MaskToChannelBasisTransform, kind)
                        transform = MaskToChannelBasisTransform(transform[1])
                else:
                    transform = self._maybe_normalization_or_regular_transform(transform)
            else:
                transform = self._construct_image_and_mask_transform(transform)
        elif self._task == 'object_detection':
            if t_(kind) == TransformKind.Transform:
                transform = self._maybe_normalization_or_regular_transform(transform)
            elif t_(kind) == TransformKind.TargetTransform:
                pass
            else:
                transform = self._construct_image_and_coco_transform(transform)

        # Add the transformation to the internal storage.
        if transform is not None:
            if prev is not None:
                self._transforms[kind].append(transform)
            else:
                self._transforms[kind] = [transform]
            self._time_inserted_transforms.append((kind, transform))

    def apply(self, contents):
        """Applies a transform to a set of input data.

        This method controls the application of the actual transforms. It
        does this inside of a context that can control the application
        of the transform and manage errors more effectively.

        The method of application is mostly similar across the different
        tasks, as the differentiation of the methods is taken care of
        in the `assign` method and each transform is converted to act
        like a generic method with simple input arguments.

        The hierarchy in which transforms are applied is:


             transform  ->  --------|
                                    |----->   dual_transform
             target_transform ->  --|

        Furthermore, image resizing takes place before any transformations
        are applied. After the transforms are applied in this order, they
        returned and if passed again, they will have a different transform
        applied to them. The state is independent of the images passed.
        """
        image, annotation = contents

        # Iterate through the different transforms.
        for (kind, transform) in self._time_inserted_transforms:
            if t_(kind) == TransformKind.Transform:
                image = self._apply_to_objects(
                    transform, (image, ), kind)
            if t_(kind) == TransformKind.TargetTransform:
                annotation = self._apply_to_objects(
                    transform, (annotation, ), kind)
            if t_(kind) == TransformKind.DualTransform:
                image, annotation = self._apply_to_objects(
                    transform, (image, annotation), kind)

        # Return the processed image and annotation.
        return image, annotation

    @staticmethod
    def _apply_to_objects(transform, contents, kind):
        """Applies the actual transformations in a context."""
        try:
            return transform(*contents)
        except Exception as e:
            default_msg = (f"Encountered an error when attempting to apply "
                           f"a transform ({transform}) of kind '{kind}' to "
                           f"objects: {contents}. See the above traceback.")

            # A specific case of exception where the image first needs
            # to be converted to a PIL image before being used in a
            # general `torchvision.transforms` pipeline.
            if "PIL" in str(e):
                raise TypeError("If using a `torchvision.transforms` pipeline "
                                "when not in PyTorch training mode, you need "
                                "to include `ToTensor()` in the pipeline.")

            # Otherwise, raise the default exception.
            raise Exception(default_msg)

    def _maybe_normalization_or_regular_transform(self, transform):
        """Dispatches to the correct single-image transform construction."""
        if isinstance(transform, tuple):
            if transform[0] == 'normalize':
                return self._build_normalization_transform(transform)
        return self._construct_single_image_transform(transform)

    def _build_normalization_transform(self, transform):
        """Constructs a normalization transform if passed.

        This is a special case for transforms passed by the `normalize_images`
        'method of the `AgMLDataLoader`, since these are treated almost as
        their own independent management system in terms of resetting or
        applying them in a different method. This is called by `assign`.
        """
        # First, we check if a normalization transform already exists
        # within the transform dict, and then we get its location.
        norm_transform_index, norm_transform_index_time = -1, -1
        try:
            for i, t in enumerate(self._transforms['transform']):
                if isinstance(t, NormalizationTransformBase):
                    norm_transform_index = i
                    break
            for i, (_, t) in enumerate(self._time_inserted_transforms):
                if isinstance(t, NormalizationTransformBase):
                    norm_transform_index_time = i
                    break
        except:
            self._transforms['transform'] = []

        if transform[1] == 'scale':
            tfm = ScaleTransform(None)
            if norm_transform_index != -1:
                self._transforms['transform'][norm_transform_index] = tfm
                self._time_inserted_transforms[norm_transform_index_time] \
                    = ('transform', tfm)
            else:
                self._transforms['transform'].append(tfm)
                self._time_inserted_transforms.append(('transform', tfm))
        elif hasattr(transform[1], 'mean') or transform[1] == 'imagenet':
            try:
                mean, std = transform[1].mean, transform[1].std
            except AttributeError:
                # Default ImageNet mean and std.
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            tfm = NormalizationTransform((mean, std))
            if norm_transform_index != -1:
                self._transforms['transform'][norm_transform_index] = tfm
                self._time_inserted_transforms[norm_transform_index_time] \
                    = ('transform', tfm)
            else:
                self._transforms['transform'].append(tfm)
                self._time_inserted_transforms.append(('transform', tfm))
        elif transform[1] == 'reset':
            if norm_transform_index != -1:
                self._transforms['transform'].pop(norm_transform_index)
                self._time_inserted_transforms.pop(norm_transform_index_time)
        return None

    # The following methods implement different checks which validate
    # as well as process input transformations, and manage the backend.
    # The transforms here will be also checked to match a specific
    # backend. Alternatively, the backend will dynamically be switched.

    @staticmethod
    def _construct_single_image_transform(transform):
        """Validates a transform which is applied to a single image.

        This is used for image classification transforms, which only
        apply to the input image, as well as other general tasks whenever
        a transform that is only applied to the image is passed, e.g.,
        a visual augmentation like random contrasting.
        """
        # This case is used for clearing a transformation.
        if transform is None:
            return None

        # A general functional transformation. We don't verify what happens in
        # the function. The only check which occurs here that the signature of
        # the function is valid. Note that this also includes partial methods.
        elif isinstance(transform, (types.FunctionType, functools.partial)):
            sig = inspect.signature(transform).parameters
            if not len(sig) == 1:
                raise TypeError("Expected a single-image transform passed "
                                "to `transform` to accept one input image, "
                                f"instead got {len(sig)} parameters.")
            return transform

        # An `albumentations` transform to be applied to the image. This
        # wraps the transform into a method which treats it as a regular
        # functional transform, e.g. no keyword arguments (for easy use).
        elif 'albumentations' in transform.__module__:
            return AlbumentationsTransformSingle(transform)

        # A set of `torchvision` transforms wrapped into a `T.Compose` object
        # or just a single transformation. This simply confirms the backend.
        elif 'torchvision' in transform.__module__:
            if get_backend() != 'torch':
                if user_changed_backend():
                    raise StrictBackendError(change = 'tf', obj = transform)
                set_backend('tf')
            return transform

        # A `tf.keras.Sequential` preprocessing model or an individual
        # Keras preprocessing layer. This simply confirms the backend.
        elif 'keras' in transform.__module__:
            if get_backend() != 'tf':
                if user_changed_backend():
                    raise StrictBackendError(change = 'torch', obj = transform)
                set_backend('torch')
            return transform

        # Otherwise, it may be a transform from a (lesser-known) third-party
        # library, in which case we just return it as a callable. Transforms
        # which are used in a more complex manner should be passed as decorators.
        return transform

    @staticmethod
    def _construct_image_and_mask_transform(transform):
        """Validates a transform for an image and annotation mask.

        This is used for a semantic segmentation transform. Such
        transformations should be passed as the following:

        - An `albumentations` transform pipeline that may include
          spatial and/or visual augmentation.
        - A method to independently or dually apply transformations
          to the image and annotation mask.
        - A `torchvision.transforms` or `tf.keras.Sequential` pipeline
          which will be applied to the image and mask using the same
          random seed, for reproducibility. Use the provided method
          `generate_keras_segmentation_dual_transform` for this.
        """
        # This case is used for clearing a transformation.
        if transform is None:
            return None

        # A general functional transformation. We don't verify what happens in
        # the function. The only check which occurs here that the signature of
        # the function is valid. Note that this also includes partial methods.
        elif isinstance(transform, (types.FunctionType, functools.partial)):
            sig = inspect.signature(transform).parameters
            if not len(sig) == 2:
                raise TypeError(f"Expected a semantic segmentation transform "
                                f"passed to `transform` to accept two args: "
                                f"an input image and an annotation mask, "
                                f"instead got {len(sig)} parameters.")
            return transform

        # An `albumentations` transform to be applied to the image. This
        # wraps the transform into a method which treats it as a regular
        # functional transform, e.g. no keyword arguments (for easy use).
        elif 'albumentations' in transform.__module__:
            return AlbumentationsTransformMask(transform)

        # If we have the case of a transform that needs to be applied to
        # both the input and the output mask simultaneously, then we wrap
        # that into a class which undertakes that behavior. This happens
        # when the signature of the input function accepts only one input
        # parameter or it belongs to `torchvision` transform (not Keras).
        if len(inspect.signature(transform).parameters) == 1:
            if 'torchvision' in transform.__module__:
                if get_backend() != 'torch':
                    if user_changed_backend():
                        raise StrictBackendError(
                            change = 'tf', obj = transform)
                    set_backend('tf')
            elif 'keras.layers' in transform.__module__:
                if get_backend() != 'tf':
                    if user_changed_backend():
                        raise StrictBackendError(
                            change = 'torch', obj = transform)
                    set_backend('torch')
                log('Got a Keras transformation for a dual image and '
                    'mask transform. If you are passing preprocessing '
                    'layers to this method, then use `agml.data.experimental'
                    '.generate_keras_segmentation_dual_transform` in order '
                    'for the random state to be applied properly.', 'warning')
            return SameStateImageMaskTransform(transform)

        # Another type of transform, most likely some form of transform
        # class. No checks are applied here, since we can't account for
        # each of the potential cases of the transformations.
        return transform

    @staticmethod
    def _construct_image_and_coco_transform(transform):
        """Validates a transform for an image and COCO JSON dictionary.

        This is used for object detection transforms. Such transformations
        should be wrapped into a method (unless they are albumentations
        transforms). The method should accept two input arguments, the image
        and the COCO JSON dictionary, and return the two respectively.
        """
        # This case is used for clearing a transformation.
        if transform is None:
            return None

        # A general functional transformation. We don't verify what happens in
        # the function. The only check which occurs here that the signature of
        # the function is valid. Note that this also includes partial methods.
        elif isinstance(transform, (types.FunctionType, functools.partial)):
            sig = inspect.signature(transform).parameters
            if not len(sig) == 2:
                raise TypeError(f"Expected a object detection transform passed "
                                f"to `transform` to accept two args: an input "
                                f"image and a COCO JSON dictionary, instead "
                                f"got {len(sig)} parameters.")
            return transform

        # An `albumentations` transform to be applied to the image. This
        # wraps the transform into a method which treats it as a regular
        # functional transform, e.g. no keyword arguments (for easy use).
        elif 'albumentations' in transform.__module__:
            return AlbumentationsTransformCOCO(transform)

        # Another type of transform, most likely some form of transform
        # class. No checks are applied here, since we can't account for
        # each of the potential cases of the transformations.
        return transform


