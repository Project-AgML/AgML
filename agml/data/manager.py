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

import numpy as np

from agml.framework import AgMLSerializable
from agml.data.object import DataObject
from agml.data.builder import DataBuilder
from agml.data.metadata import DatasetMetadata
from agml.data.managers.transforms import TransformManager
from agml.data.managers.resize import ImageResizeManager
from agml.data.managers.training import TrainingManager

from agml.utils.general import NoArgument
from agml.backend.tftorch import convert_to_batch, is_array_like
from agml.utils.random import seed_context
from agml.utils.image import consistent_shapes
from agml.utils.logging import log


class DataManager(AgMLSerializable):
    """Manages the data for a dataset loaded in an `AgMLDataLoader`.

    The `DataManager` is the core internal object which, as its name
    suggests, manages the data being used in an `AgMLDataLoader`. Upon
    instantiation of an `AgMLDataLoader`, the internal data, which is
    represented generally as image paths and corresponding annotations,
    is placed into a `DataManager` container.

    Accessing data from an `AgMLDataLoader` calls the `DataManager`,
    which takes into account all of the potential splits, transforms,
    and batching that may have been applied to the loader.

    This enables more streamlined processing and transforming of data,
    as well as parallelizing data loading as desired (in the future).
    Another way to think about the `DataManager` is it being a list
    of `DataObjects`, with extra logic for processing.
    """
    serializable = frozenset((
        'data_objects', 'resize_manager', 'accessors', 'task',
        'dataset_name', 'shuffle', 'batch_size', 'dataset_root',
        'transform_manager', 'builder', 'train_manager'))

    def __init__(self, builder, task, name, root, **kwargs):
        # Set the basic class information.
        self._task = task
        self._dataset_name = name
        self._dataset_root = root

        # Create the `DataObject`s from the `DataBuilder`.
        if not isinstance(builder, DataBuilder):
            builder = DataBuilder.from_data(
                contents = builder,
                info = DatasetMetadata(name),
                root = root)
        self._builder = builder
        self._create_objects(self._builder, task)

        # Set up the internal transform managers. These control
        # the application of transformations to the input data.
        self._transform_manager = TransformManager(task = task)
        self._resize_manager = ImageResizeManager(
            task = task, dataset = name, root = root
        )

        # The transform and resize managers are wrapped inside of a
        # `TrainingManager`, which controls the application of
        # preprocessing to the data based on the class state.
        self._train_manager = TrainingManager(
            transform_manager = self._transform_manager,
            resize_manager = self._resize_manager,
            task = task
        )

        # While we store data in the list of `DataObject`s, the actual
        # accessing of the data by index doesn't happen by directly
        # accessing that list, but instead by accessing the array of
        # indices below. This is useful for two reasons.
        #
        # It allows a much more simpler way to store and access the
        # state of the data. E.g., when splitting the data or accessing
        # it, we only need to check through this array rather than
        # searching through or shuffling the actual list of objects.
        #
        # Secondly, it makes batching data much more straightforward and
        # allows for storing representations of the data in different
        # formats, e.g., shuffling, to be done without having to interfere
        # with the actual `DataObject`s, which is much more convenient.
        self._accessors = np.arange(len(self._data_objects))

        # The following parameters store various parameters which are
        # used internally or accessed by the `AgMLDataLoader` externally.
        self._batch_size = None
        self._shuffle = kwargs.get('shuffle', True)
        self._maybe_shuffle()

    def data_length(self):
        """Calculates the length of the data based on the batching state."""
        return len(self._accessors)

    def _create_objects(self, builder, task):
        """Creates `DataObject`s from the provided content.

        Here, `content` is a dictionary mapping an an input data piece,
        an image, with its corresponding expected output, its annotation.
        """
        self._data_objects = []
        contents = builder.get_contents()
        if 'inputs' in contents.keys():
            contents = zip(tuple(contents['inputs']),
                           tuple(contents['outputs']))
        else:
            contents = contents.items()
        for content in list(contents):
            self._data_objects.append(DataObject.create(
                contents = content, task = task,
                root = self._dataset_root))

    def _maybe_shuffle(self, seed = None):
        """Wraps automatic shuffling to see if it is enabled or not."""
        if self._shuffle:
            self.shuffle(seed = seed)

    def update_train_state(self, state):
        """Updates the training state in the `TrainingManager`."""
        self._train_manager.update_state(state)

    def shuffle(self, seed = None):
        """Shuffles the contents of the `DataManager`.

        This method simply shuffles the order of the `DataObject`s
        which are stored inside this `DataManager`. Optionally, a seed
        can be provided to shuffle them inside of a specific context.
        """
        if seed is None:
            np.random.shuffle(self._accessors)
        else:
            with seed_context(seed):
                np.random.shuffle(self._accessors)

    def generate_split_contents(self, splits):
        """Generates split contents given a dictionary of the split indexes.

        This method, given a set of data split indexes, applies the indexing
        to the original content and gets a mapping of images and annotations
        which are returned back to the `AgMLDataLoader` to be constructed into
        `DataBuilder`s and wrapped into new `DataManager`s.
        """
        if self._task == 'object_detection':
            contents = np.array(list(
                self._builder.get_contents().items()), dtype = object)
        else:
            contents = np.array(list(self._builder.get_contents().items()))
        try:
            return {k: dict(contents[v]) for k, v in splits.items()}
        except IndexError:
            raise Exception(
                f"Could not generate split contents, likely due to an error "
                f"with the metadata for the dataset `{self._dataset_name}`. "
                f"Please raise this error with the AgML team.")

    def batch_data(self, batch_size):
        """Batches the data into consistent groups.

        The batched data is stored inside of this manager, as a set of
        indexes which are read and loaded when the data is accessed.
        See the information above the `_accessors` parameter above.
        """
        # If the data is already batched and a new batch size is called,
        # then update the existing batch sizes. For unbatching the data,
        # update the batch state and then flatten the accessor array.
        if self._batch_size is not None:
            try:
                self._accessors = np.concatenate(self._accessors).ravel()
            except ValueError:
                # The array is currently 0-dimensional.
                pass
        if batch_size is None or batch_size == 0:
            self._batch_size = None
            return

        # If we have a batch size of `1`, then don't do anything
        # since this doesn't really mean to do anything.
        if batch_size == 1:
            return

        # Otherwise, calculate the actual batches and the overflow
        # of the contents, and then update the accessor.
        num_splits = len(self._accessors) // batch_size
        data_items = np.array(self._accessors)
        overflow = len(self._accessors) - num_splits * batch_size
        extra_items = data_items[-overflow:]
        try:
            batches = np.array_split(
                np.array(self._accessors
                         [:num_splits * batch_size]), num_splits)
        except ValueError:
            log(f"There is less data ({len(self._accessors)}) than the provided "
                f"batch size ({batch_size}). Consider using a smaller batch size.")
            batches = [self._accessors]
        else:
            if len(extra_items) < batch_size:
                batches.append(extra_items)
        self._accessors = np.array(batches, dtype = object)
        self._batch_size = batch_size

    def assign_resize(self, image_size, method):
        """Assigns a resizing factor for the image and annotation data."""
        if image_size is None:
            image_size = 'default'
        self._resize_manager.assign(image_size, method)

    def push_transforms(self, **transform_dict):
        """Pushes a transformation to the data transform pipeline."""
        # Check if any transforms are being reset and assign them as such.
        if all(i is NoArgument for i in transform_dict):
            transform_dict = {
                'transform': 'reset',
                'target_transform': 'reset',
                'dual_transform': 'reset'
            }
        else:
            empty_keys, reset_keys = [], []
            for key, value in transform_dict.items():
                if value is NoArgument:
                    empty_keys.append(key)
                if value is None:
                    reset_keys.append(key)
            if len(empty_keys) != 0:
                for key, value in transform_dict.items():
                    if value is None:
                        transform_dict[key] = 'reset'
                    elif value is NoArgument:
                        transform_dict[key] = None

        # There is no `dual_transform` object for image classification.
        if self._task == 'image_classification':
            if transform_dict['dual_transform'] is None:
                transform_dict['dual_transform'] = 'reset'

        # Assign the transforms to the manager in order.
        for key in ['transform', 'target_transform', 'dual_transform']:
            self._transform_manager.assign(key, transform_dict[key])

    def _load_one_image_and_annotation(self, obj):
        """Loads one image and annotation from a `DataObject`."""
        return self._train_manager.apply(
            obj = obj, batch_state = self._batch_size is not None
        )

    def _load_multiple_items(self, indexes):
        """Loads multiple images and annotations from a set of `DataObject`s."""
        # Either we're getting multiple batches, or just multiple items.
        contents = []
        if self._batch_size is not None:
            for i in indexes:
                contents.append(self._load_batch(self._accessors[i]))
        else:
            for i in indexes:
                contents.append(self._load_one_image_and_annotation(
                    self._data_objects[self._accessors[i]]))
        return contents

    def _batch_multi_image_inputs(self, images):
        """Converts either a list of images or multiple input types into a batch."""
        # If the input images are just a simple batch.
        if is_array_like(images[0]):
            return convert_to_batch(images)

        # Otherwise, convert all of them independently.
        keys = images[0].keys()
        batches = {k: [] for k in keys}
        for sample in images:
            for key in sample:
                batches[key].append(sample[key])
        return {k: self._batch_multi_image_inputs(i) for k, i in batches.items()}

    def _batch_multi_output_annotations(self, annotations):
        """Converts either a list of annotations or multiple annotation types into a batch."""
        # If the output annotations are simple objects.
        if (isinstance(annotations[0], (list, np.ndarray))
            or isinstance(annotations, (list, np.ndarray))
                and isinstance(annotations[0], (int, float))):
            if not consistent_shapes(annotations):
                annotations = np.array(annotations, dtype = object)
            else:
                annotations = np.array(annotations)
            return annotations

        # For object detection, just return the COCO JSON dictionaries.
        if self._task == 'object_detection':
            return annotations

        # Otherwise, convert all of them independently.
        keys = annotations[0].keys()
        batches = {k: [] for k in keys}
        for sample in annotations:
            for key in sample:
                batches[key].append(sample[key])
        return {k: self._batch_multi_output_annotations(i) for k, i in batches.items()}

    def _load_batch(self, batch_indexes):
        """Gets a batch of data from the dataset.

        This differs from simply getting multiple pieces of data from the
        dataset, such as a slice, in that it also stacks the data together
        into a valid batch and returns it as such.
        """
        # Get the images and annotations from the data objects.
        images, annotations = [], []
        for index in batch_indexes:
            image, annotation = self._load_one_image_and_annotation(
                self._data_objects[index])
            images.append(image)
            annotations.append(annotation)

        # Attempt to create batched image arrays.
        images = self._batch_multi_image_inputs(images)

        # Attempt the same for the annotation arrays. This is more complex
        # since there are many different types of annotations, namely labels,
        # annotation masks, COCO JSON dictionaries, etc. We need to properly
        # create a batch in each of these cases.
        annotations = self._batch_multi_output_annotations(annotations)

        # Return the batches.
        return self._train_manager.make_batch(
            images = images,
            annotations = annotations
        )

    def get(self, indexes):
        """Loads and processes a piece (or pieces) of data from the dataset.

        This is the actual accessor method that performs the loading of data
        and the relevant processing as dictated by loading, image resizing,
        transform application, and other internal processing methods such as
        creating batches. This is called by the `AgMLDataLoader` to get data.
        """
        # If there is only one index and the data is not batched,
        # then we just need to return a single `DataObject`.
        if isinstance(indexes, int) and self._batch_size is None:
            return self._load_one_image_and_annotation(
                self._data_objects[self._accessors[indexes]]
            )

        # If we have a batch of images, then return the batch.
        if isinstance(indexes, int) and self._batch_size is not None:
            return self._load_batch(self._accessors[indexes])

        # Otherwise, if there are multiple indexes (e.g., an unstacked
        # slice or just a tuple of integers), then we get multiple images.
        if isinstance(indexes, (list, tuple)):
            return self._load_multiple_items(indexes)

    








