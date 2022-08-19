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

from enum import Enum

import numpy as np

from agml.framework import AgMLSerializable
from agml.utils.image import needs_batch_dim
from agml.backend.tftorch import (
    tf, torch, set_backend, get_backend,
    user_changed_backend, StrictBackendError,
    _convert_image_to_torch, is_array_like
)


class TrainState(Enum):
    NONE = None
    TF = 'tf'
    TORCH = 'torch'
    EVAL = 'eval'
    EVAL_TF = 'eval-tf'
    EVAL_TORCH = 'eval-torch'
    FALSE = False


# Shorthand form of the enum for testing explicit values.
t_ = TrainState


class TrainingManager(AgMLSerializable):
    """Controls the training state for the `AgMLDataLoader`.

    This manager is responsible for managing the backend system and
    training parameters of the `AgMLDataLoader`. In particular, it
    manages the data in different circumstances, such as when the
    data is in "train" and "eval" modes, or when it is set to be in
    a TensorFlow/PyTorch-compatible mode.

    This allows for more efficient compatibility management with the
    different backends, for instance dynamic tensor conversion and
    image formatting to account for channels_first vs. channels_last,
    as well as being able to manage whether certain preprocessing steps
    should be applied, allowing for independent train and eval modes.
    """
    serializable = frozenset((
        'transform_manager', 'resize_manager',
        'state', 'task', 'remap_hook', 'name'))

    def __init__(self, transform_manager, resize_manager, task = None):
        # Update the general parameters for the loader.
        self._task = task
        self._name = resize_manager._dataset_name

        # The `TrainingManager` is responsible for applying the
        # actual transforms, and thus controls the `TransformManager`
        # and the `ImageResizeManager` and their relevant states.
        self._transform_manager = transform_manager
        self._resize_manager = resize_manager

        # The `state` the the loader is in is variable, and can be
        # changed during program execution (e.g., between 'train'
        # and 'eval', or even into a specific backend).
        #
        # The `state` determines the actual preprocessing functions
        # that it should apply, as it controls the managers.
        # This variable tracks that state and uses it to determine
        # which steps that it should apply.
        #
        # See the `update_state()` method to see the valid states.
        self._state: "TrainState" = TrainState.NONE

        # A hook for multi-dataset loaders.
        self._remap_hook = False

    @property
    def state(self):
        """Exposes the internal state."""
        return self._state

    def update_state(self, state):
        """Updates the state of the training manager.

        This updates the state which is used to determine what preprocessing
        steps, if any, are applied to the data. Valid states include:

        1. `None`: This means that all of the transforms and image resizing
           will be applied to the data, but no automatic batching or tensor
           conversion. This is also the default state if none is specified.
        2. 'tf': This is one of the two training modes. This applies transforms,
           image resizing, as well as automatic batching and tensor conversion.
        3. 'torch': This is one of the two training modes. This applies
           transforms, image resizing, tensor conversion, as well as image
           formatting, but no automatic batching.
        4. 'eval': By default, this only enables image resizing. However, if
           the loader is set to a 'tf' or a 'torch' state, and from here it
           is converted to 'eval', then it also keeps potential tensor conversion,
           automatic batching, and image formatting.
        5. `False`: This disables all preprocessing and simply returns the raw
           loaded images and annotations.

        There are certain caveats here. Specifically, if you want to use 'eval'
        mode but maintain the tensor conversion and other related features, you
        need to use the following order of methods.

        > loader.as_keras_sequence() # or loader.as_torch_dataset()
        > loader.eval()

        If you want to completely disable preprocessing but keep the image
        resizing, then you need to first disable all preprocessing, then
        set the loader in 'eval' mode.

        > loader.as_keras_sequence() # or loader.as_torch_dataset()
        > loader.disable_preprocessing()
        > loader.eval()

        To re-enable preprocessing at this point, e.g., just the transforms
        and resizing but no tensor conversion or automatic batching, then use
        the following, which resets the train state of the loader.

        > loader.reset_processing()

        This enables the loader to track multiple states.
        """
        # Fully disable all preprocessing.
        if t_(state) == TrainState.FALSE:
            self._state = TrainState.FALSE

        # Set the correct 'eval' mode, based on the prior state.
        elif t_(state) == TrainState.EVAL:
            if self._state == TrainState.TF:
                self._state = TrainState.EVAL_TF
            elif self._state == TrainState.TORCH:
                self._state = TrainState.EVAL_TORCH
            else:
                self._state = TrainState.EVAL

        # Apply a 'tf' or 'torch' backend conversion.
        elif t_(state) == TrainState.TORCH:
            self._state = TrainState.TORCH
            if get_backend() == 'tf':
                if user_changed_backend():
                    raise StrictBackendError(
                        change = 'torch', obj = t_(state))
                set_backend('torch')
            self._resize_manager.assign('train-auto')
        elif t_(state) == TrainState.TF:
            self._state = TrainState.TF
            if get_backend() == 'torch':
                if user_changed_backend():
                    raise StrictBackendError(
                        change = 'tf', obj = t_(state))
                set_backend('tf')
            self._resize_manager.assign('train-auto')

        # Set the default conversion (`None`).
        elif t_(state) == TrainState.NONE:
            self._state = TrainState.NONE

    def _set_annotation_remap_hook(self, hook):
        """Used to modify class annotations for multi-dataset loaders."""
        self._remap_hook = hook

    def apply(self, obj, batch_state):
        """Applies preprocessing and conversions to the data contents.

        This method is responsible for actually loading and processing
        the data according to the training state, including loading the
        data, applying transforms and resizing, as well as the training
        management as described in the class. This is called by an
        enclosing `DataManager`.

        See the `TransformManager` and the `ImageResizeManager` for more
        information on the specific preprocessing applied there, and the
        `update_state()` method for more information on the training.
        """
        # Extract the raw contents from the `DataObject`.
        contents = obj.get()

        # If there is a hook to apply (for multi-dataset loaders),
        # then apply the hook before doing anything else.
        if self._remap_hook:
            contents = self._remap_hook(contents, self._name) # noqa

        # If the state is set to `False`, then just return the raw contents.
        if self._state is TrainState.FALSE:
            return contents

        # In any other case other than `False`, we resize the images.
        contents = self._resize_manager.apply(contents)

        # If we are in a training state or `None`, (so not an evaluation
        # state or `False`), then we apply the transforms to the images.
        if self._state not in [TrainState.EVAL,
                               TrainState.EVAL_TF,
                               TrainState.EVAL_TORCH]:
            contents = self._transform_manager.apply(contents)

        # If the images are not in a batch, then we convert them to tensors
        # here, otherwise, they will be converted when the batch is created.
        if not batch_state:
            contents = self._train_state_apply(contents)

        # Return the processed contents.
        return contents

    def _train_state_apply(self, contents):
        """Preprocesses the data according to the class's training state."""
        if self._state is TrainState.NONE:
            return contents
        elif self._state in [TrainState.TF, TrainState.EVAL_TF]:
            return self._tf_tensor_convert(contents, self._task)
        elif self._state in [TrainState.TORCH, TrainState.EVAL_TORCH]:
            return self._torch_tensor_convert(contents, self._task)
        return contents

    def make_batch(self, images, annotations):
        """Creates a batch of data out of processed images and annotations."""
        if self._state in [TrainState.NONE, TrainState.FALSE, TrainState.EVAL]:
            return images, annotations
        elif self._state in [TrainState.TF, TrainState.EVAL_TF]:
            return self._tf_tensor_batch_convert(
                (images, annotations), self._task)
        elif self._state in [TrainState.TORCH, TrainState.EVAL_TORCH]:
            return self._torch_tensor_batch_convert(
                (images, annotations), self._task)
        return images, annotations

    @staticmethod
    def _tf_tensor_image_convert(image):
        """Converts potential multi-image input dicts to tensors."""
        if isinstance(image, np.ndarray):
            return tf.constant(image)
        return {k: tf.constant(i) for k, i in image.items()}

    @staticmethod
    def _tf_tensor_image_batch_convert(batch):
        """Converts potential multi-image input batch dicts to tensor dicts."""
        if isinstance(batch, np.ndarray):
            return tf.stack(batch)
        return {k: tf.stack(b) for k, b in batch.items()}

    @staticmethod
    def _tf_tensor_convert(contents, task):
        """Converts contents to `tf.Tensor`s where possible."""
        # Convert the image and annotation to `tf.Tensor`s.
        image, annotation = contents
        image = TrainingManager._tf_tensor_image_convert(image)
        if task in ['image_classification',
                    'image_regression',
                    'semantic_segmentation']:
            if isinstance(annotation, (int, np.ndarray)):
                annotation = tf.constant(annotation)
            else:
                for k, v in annotation.items():
                    annotation[k] = tf.constant(v)
        elif task == 'object_detection':
            annotation = TrainingManager._tf_tensor_coco_convert(
                annotation)

        # Add a first-dimension batch to the image.
        if isinstance(image, tf.Tensor):
            if needs_batch_dim(image):
                image = tf.expand_dims(image, axis = 0)
        else:
            for k, v in image.items():
                if needs_batch_dim(v):
                    image[k] = tf.expand_dims(v, axis = 0)
        return image, annotation

    @staticmethod
    def _tf_tensor_batch_convert(contents, task):
        """Converts batch contents to `tf.Tensor`s where possible."""
        # This stacks the images and annotations together.
        images, annotations = contents
        images = TrainingManager._tf_tensor_image_batch_convert(images)
        if task == 'image_classification':
            annotations = tf.constant(annotations)
        elif task == 'image_regression':
            if isinstance(annotations, np.ndarray):
                annotations = tf.constant(annotations)
            else:
                for k, v in annotations.items():
                    annotations[k] = tf.constant(v)
        elif task == 'semantic_segmentation':
            annotations = tf.stack(annotations, axis = 0)
        elif task == 'object_detection':
            annotations = [TrainingManager._tf_tensor_coco_convert(
                a_set) for a_set in annotations]
        return images, annotations

    @staticmethod
    def _tf_tensor_coco_convert(contents):
        """Converts a COCO JSON dictionary to a `tf.Tensor`."""
        coco_tensor = {}
        for key, value in contents.items():
            coco_tensor[key] = tf.constant(value)
        return coco_tensor

    @staticmethod
    def _torch_tensor_image_convert(image):
        """Converts potential multi-image input dicts to tensors."""
        if is_array_like(image):
            return _convert_image_to_torch(image)
        return {k: _convert_image_to_torch(i) for k, i in image.items()}

    @staticmethod
    def _torch_tensor_image_batch_convert(batch):
        """Converts potential multi-image input batch dicts to tensor dicts."""
        if is_array_like(batch):
            return torch.stack([
                _convert_image_to_torch(image) for image in batch])
        return {k: TrainingManager.
            _torch_tensor_image_batch_convert(b) for k, b in batch.items()}

    @staticmethod
    def _torch_tensor_convert(contents, task):
        """Converts contents to `torch.Tensor`s where possible."""
        image, annotation = contents
        image = TrainingManager._torch_tensor_image_convert(image)
        if task in ['image_classification',
                    'image_regression']:
            if isinstance(annotation, (int, np.ndarray)):
                annotation = torch.tensor(annotation)
            else:
                for k, v in annotation.items():
                    annotation[k] = torch.tensor(v)
        elif task == 'semantic_segmentation':
            annotation = _convert_image_to_torch(annotation)
        elif task == 'object_detection':
            annotation = TrainingManager._torch_tensor_coco_convert(
                annotation)
        return image, annotation

    @staticmethod
    def _torch_tensor_batch_convert(contents, task):
        """Converts batch contents to `torch.Tensor`s where possible."""
        images, annotations = contents
        images = TrainingManager._torch_tensor_image_batch_convert(images)
        if task in ['image_classification',
                    'image_regression']:
            if isinstance(annotations, (int, np.ndarray)):
                annotations = torch.tensor(annotations)
            else:
                for k, v in annotations.items():
                    annotations[k] = torch.tensor(v)
        elif task == 'semantic_segmentation':
            annotations = torch.stack([
                _convert_image_to_torch(a) for a in annotations])
        elif task == 'object_detection':
            annotations = [TrainingManager._torch_tensor_coco_convert(
                a_set) for a_set in annotations]
        return images, annotations

    @staticmethod
    def _torch_tensor_coco_convert(contents):
        """Converts a COCO JSON dictionary to a `torch.Tensor`."""
        coco_tensor = {}
        for key, value in contents.items():
            if key == 'segmentation':
                value = np.empty(0)
            if not isinstance(value, torch.Tensor):
                coco_tensor[key] = torch.tensor(value)
            else:
                coco_tensor[key] = value
        return coco_tensor






