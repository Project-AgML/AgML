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
import collections
from typing import Union
from decimal import Decimal, getcontext

import numpy as np

from agml.framework import AgMLSerializable
from agml.data.metadata import DatasetMetadata
from agml.data.loader import AgMLDataLoader
from agml.utils.general import (
    resolve_list_value, NoArgument
)
from agml.utils.random import seed_context, inject_random_state
from agml.utils.image import consistent_shapes
from agml.utils.logging import log
from agml.backend.tftorch import (
    get_backend, set_backend,
    user_changed_backend, StrictBackendError,
    is_array_like, convert_to_batch
)


class CollectionWrapper(AgMLSerializable):
    """Wraps a collection of items and calls their attributes and methods."""
    serializable = frozenset(('collection', 'keys'))

    def __init__(self, collection, keys = None, ignore_types = False):
        self._collection = collection
        if not ignore_types:
            if not all(isinstance(c, type(collection[0])) for c in collection):
                raise TypeError(
                    f"Items in a collection should all be of the "
                    f"same type, got {[type(i) for i in collection]}")
        self._keys = keys

    def __len__(self):
        return len(self._collection)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._collection[self._keys.index(item)]
        return self._collection[item]

    @property
    def keys(self):
        return self._keys

    def get_attributes(self, attr):
        return [getattr(c, attr) for c in self._collection]

    def call_method(self, method, args = None, kwargs = None):
        if kwargs is None:
            kwargs = {}
        if args is None:
            args = ()
        elif isinstance(args, tuple):
            if not len(args) == len(self._collection):
                raise IndexError(
                    f"Got {len(args)} unique arguments for a "
                    f"collection of length {len(self._collection)}.")
            return [getattr(c, method)(arg, **kwargs)
                    for c, arg in zip(self._collection, args)]
        return [getattr(c, method)(*args, **kwargs) for c in self._collection]

    def apply(self, method, args = None):
        if args is None:
            for c in self._collection:
                method(c)
        else:
            for c, arg in zip(self._collection, args):
                method(c, arg)


class MultiDatasetMetadata(AgMLSerializable):
    """Stores metadata for a collection of AgML datasets.

    Functionally, this class is just a wrapper around multiple
    `DatasetMetadata` objects, and for each of the traditional
    attributes in the original `DatasetMetadata`, returns a
    dictionary with all of the values for the corresponding
    datasets rather than just a single value.
    """
    serializable = frozenset(("names", "metas", "task"))

    def __init__(self, datasets):
        # Build a collection of metadata objects.
        self._names = datasets
        self._metas = CollectionWrapper(
            [DatasetMetadata(d) for d in datasets])
        self._validate_tasks()
        
    @classmethod
    def _from_collection(cls, metas):
        """Instantiates a `MultiDatasetMetadata` object from a collection."""
        obj = MultiDatasetMetadata.__new__(MultiDatasetMetadata)
        obj._names = [meta.name for meta in metas]

        # We need to ignore types since some metadata objects might be
        # regular `DatasetMetadata`, but some might be `CustomDatasetMetadata`.
        obj._metas = CollectionWrapper(metas, ignore_types = True)
        obj._validate_tasks()
        return obj

    def _validate_tasks(self):
        # For a collection of datasets to work, they all need
        # to be of the same task. This is a check of this.
        tasks = self._metas.get_attributes('tasks')
        if not all(tasks[0].ml == t.ml for t in tasks):
            raise ValueError("To use a collection of datasets, all of them "
                             f"must be of the same task. Got tasks {tasks} "
                             f"for the provided datasets {self._names}.")
        self._task = tasks[0]

    def __getattr__(self, attr):
        # This is the main functionality of the `DatasetMetadata`
        # class. Rather than re-writing each of the attributes,
        # this method simply checks whether the requested attribute
        # or method exists in the original `DatasetMetadata` class,
        # then calls the relevant method on each of the stored
        # metadata objects and returns them.
        if hasattr(DatasetMetadata, attr):
            obj = getattr(DatasetMetadata, attr)

            # If it is a property, then we need to return something.
            if isinstance(obj, property):
                return {k: v for k, v in zip(
                    self._names, self._metas.get_attributes(attr))}

            # If it is a method, then it is just printing.
            if isinstance(obj, types.FunctionType):
                self._metas.call_method(attr)
                return lambda: None # To act like a function.

        # Otherwise, raise an error for an invalid argument.
        else:
            raise AttributeError(f"Invalid attribute {attr} for `DatasetMetadata`.")


class AnnotationRemap(AgMLSerializable):
    """A helper class to remap annotation labels for multiple datasets."""
    serializable = frozenset((
        "general_class_to_num", "num_to_class", "task",
        "generalize_class_detections"))
    
    def __init__(self, general_class_to_num, num_to_class,
                 task, generalize_class_detections = False):
        self._task = task
        self._num_to_class = num_to_class
        self._general_class_to_num = general_class_to_num
        self._generalize_class_detections = generalize_class_detections
        
    def __call__(self, contents, name):
        """Re-maps the annotation for the new, multi-dataset mapping."""
        image, annotations = contents

        # For image classification, simply re-map the label number.
        if self._task == 'image_classification':
            annotations = self._general_class_to_num[
                self._num_to_class[name][annotations].lower()]

        # For semantic segmentation, re-map the mask IDs.
        if self._task == 'semantic_segmentation':
            unique_values = np.unique(annotations)[1:] # skip background
            new_values = np.array([self._general_class_to_num[
                self._num_to_class[name][c].lower()]
                                for c in unique_values])
            for u, n in zip(unique_values, new_values):
                annotations[np.where(annotations == u)[0]] = n

        # For object detection, also just remap the annotation ID.
        if self._task == 'object_detection':
            if self._generalize_class_detections:
                annotations['category_id'] = np.ones_like(
                    annotations['category_id'])
            else:
                category_ids = annotations['category_id']
                category_ids[np.where(category_ids == 0)[0]] = 1 # fix
                new_ids = np.array([self._general_class_to_num[
                    self._num_to_class[name][c].lower()]
                                    for c in category_ids])
                annotations['category_id'] = new_ids
        return image, annotations


class AgMLMultiDatasetLoader(AgMLSerializable):
    """Loads and holds a collection of multiple datasets.

    This class serves as an interface for the `AgMLDataLoader` when
    using multiple datasets together, and enables similar seamless
    usage as with the traditional `AgMLDataLoader`, with additional
    features to support the usage of multiple datasets.

    Functionally, this class acts as a wrapper around multiple
    `AgMLDataLoader` objects, and draws upon similar functionality
    to the `DataManager` in order to access data from multiple objects.
    """
    serializable = frozenset(
        ('info', 'loaders', 'loader_accessors', 'class_meta',
         'set_to_keys', 'bounds', 'batch_size', 'shuffle_data',
         'data_distributions', 'is_split', 'train_data',
         'val_data', 'test_data'))

    def __init__(self, datasets, **kwargs):
        """Instantiates an `AgMLDataLoader` with multiple datasets."""
        # The order of the dataset images and classes should be
        # agnostic to the order which they are passed in. So, the
        # same datasets but in a different order should yield the
        # same class mapping and internal representation. We sort
        # the datasets in order to ensure the same order every time.
        datasets = sorted(datasets)

        # Set up the datasets and their associated metadata. Since
        # there are multiple datasets, the `info` parameter here
        # will return a `MultipleDatasetMetadata` object which
        # is functionally similar to the `DatasetMetadata` object,
        # but returns dictionaries with the relevant parameters for
        # each of the different datasets, rather than just for one.
        self._info = MultiDatasetMetadata(datasets)

        # Create a set of `AgMLDataLoader` objects for each of the
        # datasets which are provided to the loader. These will
        # be the primary access interface for the objects.
        self._make_loaders(datasets, **kwargs)

        # Adapt all of the individual class types in the sub-datasets,
        # such that if there are two classes which are the same, then
        # the `class` that they represent should be the same.
        self._adapt_classes()

        # Similar to how the internal `DataManager` works, this wrapper
        # will access data from the internal `AgMLDataLoader` using an
        # accessor array, which will be the length of all of the datasets
        # in the loader combined. It will go the length of the datasets
        # in the order provided, where index 0 represents the first item
        # in the first dataset, and index 1 represents the last item in
        # the last dataset. Batching works in a similar way.
        self._loader_accessors = np.arange(
            sum(v for v in self._info.num_images.values()))
        sets = self._info.num_images.keys()
        bounds = np.cumsum(list(self._info.num_images.values())).tolist()
        bounds = (0, ) + (*bounds, )
        self._set_to_keys = {}
        self._bounds = {s: b for s, b in zip(sets, bounds)}
        for i, set_ in enumerate(sets):
            value = 0 if i == 0 else 1
            self._set_to_keys.update(dict.fromkeys(
                np.arange(bounds[i] - value,
                          bounds[i + 1] + 1), set_))

        # The batch size is modified similarly like the `DataManager`.
        # Since all of the data loaders should have the same properties,
        # transforms, and other values, we get the `make_batch` method
        # from the first loader and bound that to make batches.
        self._batch_size = None
        self._make_batch = self._loaders[0]._manager._train_manager.make_batch

        # Shuffling in a multi-dataset loader takes place on a high-level.
        # The contents of the actual datasets themselves are not shuffled.
        # Instead, the accessor array is shuffled (like the DataManager).
        self._shuffle_data = kwargs.get('shuffle', True)
        if self._shuffle_data:
            self.shuffle(self._shuffle_data)

        # We need to transform the class annotations before they are sent
        # through any other transforms, since those other transforms may
        # modify the value of the annotation and potentially cause issues
        # with the output annotation. However, we need this to happen
        # even when transforms are disabled. So, we modify each of the
        # loaders' `TrainingManager`s with a special argument to call an
        # extra helper method which in turn modifies the class annotation
        # before any other transforms are applied, and in any case.
        self._loaders.apply(
            lambda x: x._manager._train_manager._set_annotation_remap_hook(
                AnnotationRemap(
                    self.class_to_num, self._info.num_to_class, self.task)))

        # We can't use the `auto` resizing mode for the resizing manager,
        # because of the complexity of trying to make it work with multiple
        # different data types. So, disable auto mode to prevent errors.
        self._loaders.apply(
            lambda x: x._manager._resize_manager.disable_auto())

        # The data is not split to begin with. So, we set the split
        # parameter to false and store all of the split datasets themselves
        # as empty variables (which will be updated if and when it is split).
        self._is_split = False
        self._train_data = None
        self._val_data = None
        self._test_data = None

        # Calculate the distribution of the data.
        self._data_distributions = self._info.num_images

        # Total number of images in the entire dataset.
        self._num_images = sum(self._info.num_images.values())
        
    @classmethod
    def _instantiate_from_collection(cls, *loaders, classes):
        """Instantiates an `AgMLMultiDatasetLoader` directly from a collection.
        
        This method is, in essence, a wrapper around the actual `__init__`
        method for the multi-loader, but one which takes into account the fact
        that the loaders are already instantiated, and thus works around those
        already-provided parameters, rather than starting from scratch. 
        """
        obj = AgMLMultiDatasetLoader.__new__(AgMLMultiDatasetLoader)
        
        # Create the custom dataset metadata wrapper.
        obj._info = MultiDatasetMetadata._from_collection([
            loader.info for loader in loaders])
        
        # Add the loaders and adapt classes.
        obj._loaders = CollectionWrapper(
            loaders, keys = [loader.info.name for loader in loaders])
        obj._adapt_classes(cls = classes)
        
        # The remaining contents here are directly copied from the above
        # `__init__` method, without comments (see above for information):
        # Construct the accessor array.
        obj._loader_accessors = np.arange(
            sum(v for v in obj._info.num_images.values()))
        sets = obj._info.num_images.keys()
        bounds = np.cumsum(list(obj._info.num_images.values())).tolist()
        bounds = (0, ) + (*bounds, )
        obj._set_to_keys = {}
        obj._bounds = {s: b for s, b in zip(sets, bounds)}
        for i, set_ in enumerate(sets):
            value = 0 if i == 0 else 1
            obj._set_to_keys.update(dict.fromkeys(
                np.arange(bounds[i] - value,
                          bounds[i + 1] + 1), set_))
            
        # Set the batch size and shuffling.
        obj._batch_size = None
        obj._make_batch = obj._loaders[0]._manager._train_manager.make_batch
        obj._shuffle_data = loaders[0].shuffle_data
        if obj._shuffle_data:
            obj.shuffle(obj._shuffle_data)
            
        # Transform annotations and resizing.
        obj._loaders.apply(
            lambda x: x._manager._train_manager._set_annotation_remap_hook(
                AnnotationRemap(
                    obj.class_to_num, obj._info.num_to_class, obj.task)))
        obj._loaders.apply(
            lambda x: x._manager._resize_manager.disable_auto())
        
        # Finalize parameters.
        obj._is_split = False
        obj._train_data = None
        obj._val_data = None
        obj._test_data = None
        obj._data_distributions = obj._info.num_images
        obj._num_images = sum(obj._info.num_images.values())

        # Return the object.
        return obj

    def __len__(self):
        # Return the length of the data, subject to batching.
        return self._data_length()
            
    def __getitem__(self, indexes: Union[int, slice, tuple, list]):
        # The `__getitem__` logic adopts the `DataManager` approach
        # towards getting multiple items, wrapped into this class.
        if isinstance(indexes, slice):
            data = np.arange(self._data_length())
            indexes = data[indexes].tolist()
        if isinstance(indexes, int):
            indexes = [indexes]
        for idx in indexes:
            if idx not in range(len(self)):
                raise IndexError(
                    f"Index {idx} out of range of "
                    f"AgMLDataLoader length: {len(self)}.")
        return self._get_item_impl(resolve_list_value(indexes))

    def __iter__(self):
        for indx in range(len(self)):
            yield self[indx]

    def __repr__(self):
        dsp = ", "
        out = f"<AgMLDataLoader: (datasets=[{dsp.join(self._info.name)}]"
        out += f", task={self.task}"
        out += f") at {hex(id(self))}>"
        return out

    def __str__(self):
        return repr(self)

    def copy(self):
        """Returns a deep copy of the data loader's contents."""
        return self.__copy__()

    def __copy__(self):
        """Copies the loader and updates its state."""
        cp = super(AgMLMultiDatasetLoader, self).__copy__()
        cp.copy_state(self)
        return cp

    def copy_state(self, loader):
        """Copies the state of another `AgMLDataLoader` into this loader.

        This method copies the state of another `AgMLDataLoader` into this
        loader, including its transforms, resizing, and training state. Other
        general parameters such as batch size and shuffling are left intact.

        Parameters
        ----------
        loader : AgMLDataLoader
            The data loader from which the state should be copied.

        Returns
        -------
        This `AgMLDataLoader`.
        """
        # Copy the state for all the sub-loaders. If the loader state is of
        # a multi-loader, then only copy the state of its first loader.
        if isinstance(loader, AgMLMultiDatasetLoader):
            loader = loader._loaders[0]
        self._loaders.apply(
            lambda x: x.copy_state(loader)
        )

    def _make_loaders(self, datasets, **kwargs):
        """Constructs the loaders for the datasets in the collection."""
        # Get and validate the `dataset_path` argument.
        if 'dataset_path' in kwargs:
            dataset_path = kwargs.get('dataset_path')
            if isinstance(dataset_path, collections.Sequence):
                if not len(datasets) == len(dataset_path):
                    raise IndexError(
                        f"Got a sequence for the `dataset_path` of a "
                        f"multi-dataset `AgMLDataLoader`, but it is not "
                        f"the same length as the number of datasets: "
                        f"{len(datasets)} datasets ({datasets}) but "
                        f"{len(dataset_path)} paths ({dataset_path}).")
            elif isinstance(dataset_path, str):
                dataset_path = [dataset_path] * len(datasets)
        else:
            dataset_path = False
        kwargs.update({'dataset_path': dataset_path})

        # Create all of the loaders.
        self._loaders = CollectionWrapper([
            AgMLDataLoader(dataset, **kwargs) for dataset in datasets],
            keys = datasets)

    def _adapt_classes(self, cls = None):
        """Adapts the classes in the loader."""
        # Get all of the unique classes in the loader.
        classes = self._info.classes.values()
        class_values = [[o.lower() for o in c] for c in classes]
        class_values = [i for s in class_values for i in s]
        unique_classes = np.unique(class_values).tolist()

        # Check that they match the given classes, if such a list is passed.
        if cls is not None:
            if not set(cls) == set(unique_classes): # noqa
                raise ValueError(
                    f"Given list of classes {cls} to `AgMLDataLoader.merge`, "
                    f"but calculated classes {unique_classes}. Check that the "
                    f"given classes match the actual classes in the given datasets.")
            unique_classes = cls

        # Create a class metadata storing all of the unique
        # classes belonging to this loader and their mappings.
        self._class_meta = {
            'classes': unique_classes,
            'num_classes': len(unique_classes),
            'class_to_num': {
                v: k + 1 for k, v in enumerate(unique_classes)},
            'num_to_class': {
                k + 1: v for k, v in enumerate(unique_classes)}
        }

    def _data_length(self):
        """Calculates the length of the data from the different datasets."""
        return len(self._loader_accessors)

    @property
    def info(self):
        """Returns a `DatasetMetadata` object containing dataset info.

        The contents returned in the `DatasetMetadata` object can be used
        to inspect dataset metadata, such as the location the data was
        captured, the data formats, and the license/copyright information.
        See the `DatasetMetadata` class for more information.
        """
        return self._info

    @property
    def task(self):
        """Returns the ML task that this dataset is constructed for."""
        return self._info._task.ml

    @property
    def num_images(self):
        """Returns the number of images in the entire dataset."""
        return sum(self.data_distributions.values())

    @property
    def classes(self):
        """Returns the classes that the dataset is predicting."""
        return self._class_meta['classes']

    @property
    def num_classes(self):
        """Returns the number of classes in the dataset."""
        return self._class_meta['num_classes']

    @property
    def num_to_class(self):
        """Returns a mapping from a number to a class label."""
        return self._class_meta['num_to_class']

    @property
    def class_to_num(self):
        """Returns a mapping from a class label to a number."""
        return self._class_meta['class_to_num']

    @property
    def data_distributions(self):
        """Returns a distribution of data in the loader.

        This property returns a dictionary which provides information
        about the data within the loader; e.g., how many images are
        from each of the datasets in the collection.
        """
        return self._data_distributions

    def _generate_split_loader(self, loaders, split):
        """Generates a split `AgMLDataLoader`."""
        # Check if the data split exists.
        if any(loader is None for loader in loaders):
            raise ValueError(
                f"Attempted to access split loader {split} when "
                f"parent loader has not been split.")

        # Create a new `CollectionWrapper` around the datasets.
        new_collection = CollectionWrapper(
            loaders, keys = [l_.name for l_ in loaders])

        # Get the state of the current loader and update it
        # with the new collection of loaders. Then, update
        # the accessors and number of images with the newly
        # reduced quantity (due to the splitting of data).
        loader_state = self.copy().__getstate__()
        loader_state['loaders'] = new_collection
        total_num_images = sum(len(loader) for loader in loaders)
        data_distributions = {
            loader.name: len(loader) for loader in loaders}
        loader_state['data_distributions'] = data_distributions
        accessors = np.arange(0, total_num_images)
        if self._shuffle_data:
            np.random.shuffle(accessors)
        loader_state['loader_accessors'] = accessors
        batch_size = loader_state.pop('batch_size')
        loader_state['batch_size'] = None

        # Re-generate the mapping for bounds.
        sets = self._info.num_images.keys()
        bound_ranges = np.cumsum([len(loader) for loader in loaders]).tolist()
        bound_ranges = (0, ) + (*bound_ranges, )
        set_to_keys = {}
        bounds = {s: b for s, b in zip(sets, bound_ranges)}
        for i, set_ in enumerate(sets):
            value = 0 if i == 0 else 1
            set_to_keys.update(dict.fromkeys(
                np.arange(bound_ranges[i],
                          bound_ranges[i + 1] + value), set_))
        loader_state['set_to_keys'] = set_to_keys
        loader_state['bounds'] = bounds

        # Create the new loader from the updated state.
        new_loader = AgMLMultiDatasetLoader.__new__(AgMLMultiDatasetLoader)
        new_loader.__setstate__(loader_state)

        # Batching data should be re-done independently.
        if batch_size is not None:
            new_loader.batch(batch_size = batch_size)

        # Block out all of the splits of the already split
        # loader and set the `_is_split` attribute to True,
        # preventing future splits, and return.
        for attr in ['train', 'val', 'test']:
            setattr(new_loader, f'_{attr}_data', None)
        new_loader._is_split = True
        return new_loader

    @property
    def train_data(self):
        """Stores the `train` split of the data in the loader."""
        if isinstance(self._train_data, AgMLMultiDatasetLoader):
            return self._train_data
        self._train_data = self._generate_split_loader(
            self._loaders.get_attributes('train_data'), split = 'train')
        return self._train_data

    @property
    def val_data(self):
        """Stores the `val` split of the data in the loader."""
        if isinstance(self._val_data, AgMLMultiDatasetLoader):
            return self._val_data
        self._val_data = self._generate_split_loader(
            self._loaders.get_attributes('val_data'), split = 'val')
        return self._val_data

    @property
    def test_data(self):
        """Stores the `test` split of the data in the loader."""
        if isinstance(self._test_data, AgMLMultiDatasetLoader):
            return self._test_data
        self._test_data = self._generate_split_loader(
            self._loaders.get_attributes('test_data'), split = 'test')
        return self._test_data

    def eval(self):
        """Sets the `AgMLDataLoader` in evaluation mode.

        Evaluation mode disables transforms, and only keeps the loader applying
        resizing to the contents. If the loader was previously set into TensorFlow
        or PyTorch mode, however, it will also keep up tensor conversion and
        potential batch adding (see `as_keras_sequence()` and `as_torch_dataset()`
        methods for more information on the exact operations).

        This method does not completely disable preprocessing, to completely
        disable preprocessing, use `loader.disable_preprocessing()`. Additionally,
        if you want to keep only the resizing but not the implicit tensor
        conversions based on the backend, then run:

        > loader.disable_preprocessing() # or loader.reset_preprocessing()
        > loader.eval()

        This will refresh the backend conversions and return it to `eval` mode.

        Returns
        -------
        The `AgMLDataLoader` object.
        """
        self._loaders.call_method('eval')
        return self

    def disable_preprocessing(self):
        """Disables all preprocessing on the `AgMLDataLoader`.

        This sets the loader in a no-preprocessing mode (represented internally as
        `False`), where only the raw data is returned: no transforms, resizing, or
        any conversion to any type of backend. This can be used to test or inspect
        the original data contents of the loader before processing.

        The loader can be set into any mode from here, for instance see `eval()`,
        `as_keras_sequence()`, and `as_torch_dataset()` for specific examples on
        the different potential training and evaluation states. If you just want
        to reset the loader to its default state, which applies only transforms
        and resizing, then use `loader.reset_preprocessing()`.

        Returns
        -------
        The `AgMLDataLoader` object.
        """
        self._loaders.call_method('disable_preprocessing')
        return self

    def reset_preprocessing(self):
        """Re-enables preprocessing on the `AgMLDataLoader`.

        This resets the loader back to its default train state, namely where it
        applies just the given transforms and content resizing. This is a consistent
        method, meaning that regardless of the prior train state of the loader
        before running this method, it will hard reset it to its original state
        (similar to `disable_preprocessing()`, but it keeps some preprocessing).

        Returns
        -------
        The `AgMLDataLoader` object.
        """
        self._loaders.call_method('reset_preprocessing')
        return self

    def on_epoch_end(self):
        """Shuffles the dataset on the end of an epoch for a Keras sequence.

        If `as_keras_sequence()` is called and the `AgMLDataLoader` inherits
        from `tf.keras.utils.Sequence`, then this method will shuffle the
        dataset on the end of each epoch to improve training.
        """
        self._loaders.call_method('on_epoch_end')

    def as_keras_sequence(self):
        """Sets the `DataLoader` in TensorFlow mode.

        This TensorFlow extension converts the loader into a TensorFlow mode,
        adding inheritance from the superclass `keras.utils.Sequence` to enable
        it to be used directly in a Keras pipeline, and adding extra preprocessing
        to the images and annotations to make them compatible with TensorFlow.

        The main features added on enabling this include:

        1. Conversion of output images and annotations to `tf.Tensor`s.
        2. Adding an implicit batch size dimension to images even when the
           data is not batched (for compatibility in `Model.fit()`).
        3. Adding inheritance from `keras.utils.Sequence` so that any
           `AgMLDataLoader` object can be used directly in `Model.fit()`.
        4. Setting the data loader to use a constant image shape, namely
           `auto` (which will default to (512, 512) if none is found).
           This can be overridden by manually setting the image shape
           parameter back after running this method. Note that this may
           result in errors when attempting implicit tensor conversion.

        Returns
        -------
        The `AgMLDataLoader` object.
        """
        self._loaders.call_method('as_keras_sequence')
        return self

    def as_torch_dataset(self):
        """Sets the `DataLoader` in PyTorch mode.

        This PyTorch extension converts the loader into a PyTorch mode, adding
        inheritance from th superclass `torch.utils.data.Dataset` to enable it to
        be used directly in a PyTorch pipeline, and adding extra preprocessing to
        the images and annotations to make them compatible with PyTorch.

        The main features added on enabling this include:

        1. Conversion of output images and annotations to `torch.Tensor`s.
        2. Converting the channel format of the input images from the default,
           channels_last, into channels_first (NHWC -> NCHW).
        3. Adding inheritance from `torch.utils.data.Dataset` so that any
           `AgMLDataLoader` object can be used with a `torch.utils.data.DataLoader`.
        4. Setting the data loader to use a constant image shape, namely
           `auto` (which will default to (512, 512) if none is found).
           This can be overridden by manually setting the image shape
           parameter back after running this method. Note that this may
           result in errors when attempting implicit tensor conversion.

        Returns
        -------
        The `AgMLDataLoader` object.
        """
        self._loaders.call_method('as_torch_dataset')

    @property
    def shuffle_data(self):
        """Returns whether the loader is set to shuffle data or not.

        By default, if no value is passed in initialization, this is set to
        `True`. It can be manually toggled to `False` using this property.
        """
        return self._shuffle_data

    @shuffle_data.setter
    def shuffle_data(self, value):
        """Set whether the loader should shuffle data or not.

        This can be used to enable/disable shuffling, by passing
        either `True` or `False`, respectively.
        """
        if not isinstance(value, bool):
            raise TypeError("Expected either `True` or `False` for 'shuffle_data'.")
        self._shuffle_data = value

    def shuffle(self, seed = None):
        """Potentially shuffles the contents of the loader.

        If shuffling is enabled on this loader (`shuffle = False` has
        not been passed to the instantiation), then this method will
        shuffle the order of contents in it. A seed can be provided to
        shuffle the dataset to an expected order.

        If the data is already batched, then the batch contents will be
        shuffled. For instance, if we have data batches [[1, 2], [3, 4]],
        then the shuffling result will be [[3, 4], [1, 2]]. If you want
        all of the contents to be shuffled, call `shuffle` before batching.

        Note that the data is automatically shuffled upon instantiation,
        unless the `shuffle = False` parameter is passed at instantiation.
        However, this disables automatic shuffling for the class
        permanently, and this method must be called to shuffle the data.

        Parameters
        ----------
        seed : int, optional
            A pre-determined seed for shuffling.

        Returns
        -------
        The `AgMLDataLoader` object.
        """
        if seed is None:
            np.random.shuffle(self._loader_accessors)
        else:
            with seed_context(seed):
                np.random.shuffle(self._loader_accessors)
        return self

    def take_dataset(self, name) -> "AgMLDataLoader":
        """Takes one of the datasets in the multi-dataset collection.

        This method selects one of the datasets (as denoted by `name`)
        in this multi-dataset collection and returns an `AgMLDataLoader`
        with its contents. These contents will be subject to any transforms
        and modifications as applied by the main loader, but the returned
        loader will be a copy, such that any new changes made to the main
        multi-dataset loader will not affect the new loader.

        Parameters
        ----------
        name : str
            The name of one of the sub-datasets of the loader.

        Returns
        -------
        An `AgMLDataLoader`.
        """
        return self._loaders[name].copy()

    @inject_random_state
    def take_random(self, k) -> "AgMLMultiDatasetLoader":
        """Takes a random set of contents from the loader.

        This method selects a sub-sample of the contents in the loader,
        based on the provided number of (or proportion of) elements `k`.
        It then returns a new loader with just this reduced number of
        elements. The new loader is functionally similar to the original
        loader, and contains all of the transforms/batching/other settings
        which have been applied to it up until this method is called.

        Note that the data which is sampled as part of this new loader
        is not removed from the original loader; this simply serves as an
        interface to use a random set of images from the full dataset.

        For a multi-dataset loader, data is sampled from each of the
        sub-datasets proportionally, e.g., the proportion of images
        in the new dataset (per each sub-dataset) will be the same as
        in the original dataset.

        Parameters
        ----------
        k : {int, float}
            Either an integer specifying the number of samples or a float
            specifying the proportion of images from the total to take.

        Returns
        -------
        A reduced `AgMLDataLoader` with the new data.
        """
        # Parse the input to an integer.
        if isinstance(k, float):
            # Check that 0.0 <= k <= 1.0.
            if not 0.0 <= k <= 1.0:
                raise ValueError(
                    "If passing a proportion to `take_class`, "
                    "it should be in range [0.0, 1.0].")

            # Convert the proportion float to an absolute int. Note that
            # the method used is rounding up to the nearest int for cases
            # where there is not an exact proportional equivalent.
            getcontext().prec = 4  # noqa
            proportion = Decimal(k) / Decimal(1)
            num_images = self.num_images
            k = int(proportion * num_images)

        # If the input is an integer (or the float is converted to an int
        # above), then select a random sampling of images from the dataset.
        if isinstance(k, int):
            # Check that `k` is valid for the number of images in the dataset.
            if not 0 <= k <= self.num_images:
                raise ValueError(
                    f"Received a request to take a random sampling of "
                    f"{k} images, when the dataset has {self.num_images}.")

            # Calculate the proportions. If the total sum is less than `k`,
            # add 1 to the dataset with the lowest number of images.
            getcontext().prec = 4  # noqa
            num_images = self.num_images
            proportions = {key: int((Decimal(val) / Decimal(num_images)) * k)
                            for key, val in self._data_distributions.items()}
            if sum(proportions.values()) != num_images:
                diff = sum(proportions.values()) - k
                smallest_split = list(proportions.keys())[
                    list(proportions.values()).index(
                        min(proportions.values()))]
                proportions[smallest_split] = proportions[smallest_split] - diff
            return self._generate_split_loader(
                self._loaders.call_method(
                    'take_random', tuple(proportions.values())), 'train')

        # Otherwise, raise an error.
        else:
            raise TypeError(
                f"Expected only an int or a float when "
                f"taking a random split, got {type(k)}.")

    @inject_random_state
    def split(self, train = None, val = None, test = None, shuffle = True):
        """Splits the data into train, val and test splits.

        For this multi-dataset loader, an even split of data will
        be selected from each dataset. E.g., if you have a loader
        of two datasets, each with 100 images, and want a train/test
        split of 0.9/0.1, then 90 images from each dataset will form the
        new training set, and 10 images from each will form the new
        test set. This is to ensure consistency in training.

        By default, this method does nothing (or if the data has been
        split into sets, it resets them all to one set). Setting the
        `train`, `val`, and `test` parameters randomly divides the
        data into train, validation, and/or test sets, depending on
        which ones are provided and their values.

        Values can either be passed as exact numbers or as proportions,
        e.g. either `train = 80, test = 20` in a 100-value dataset, or
        as `train = 0.8, test = 0.2`. Whichever value is not passed,
        e.g. `val` in this case, has no value in the loader.

        Parameters
        ----------
        train : {int, float}
            The split for training data.
        val : {int, float}
            The split for validation data.
        test : {int, float}
            The split for testing data.
        shuffle : bool
            Whether to shuffle the split data.

        Notes
        -----
        Any processing applied to this `AgMLDataLoader` will also be present
        in the split loaders until they are accessed from the class. If you
        don't want these to be applied, access them right after splitting.
        """
        # We run this by applying dataset splitting to all of the individual
        # loaders, which then create and get their own even split of data
        # for the dataset they are loading. This will first ensure an even
        # split of data from each of the different datasets.
        #
        # Similar to the `AgMLDataLoader` itself, we will not access the
        # `train/val/test_data` parameters instantly, as this will allow
        # any new transforms or other parameters which are applied to the
        # parent loader to be also applied to all of the child split loaders
        # until they are actually accessed in this overhead multi-dataset
        # loader class. Then, they will be unset and created.
        self._loaders.apply(
            lambda x: x.split(
                train = train, val = val, test = test, shuffle = shuffle))

    def batch(self, batch_size = None):
        """Batches sets of image and annotation data according to a size.

        This method will group sets of data together into batches of size
        `batch_size`. In turn, items gathered from the loader will, rather
        than being an image and annotation, be an array of images and an
        array of annotations (not an array of image/annotation pairs).

        Batching data will include a `batch` dimension for the images and
        annotations that are returned (e.g., the image array will have
        dimensions NHWC instead of HWC). If the data is not batched, this
        dimension will not be present unless the loader is in training mode.

        The data can be un-batched by passing `None` to batch size (or
        calling the method with no arguments).

        Parameters
        ----------
        batch_size : int, None
            The number of groups to batch data together into.

        Notes
        -----
        The last batch will be of size <= `batch_size`.
        """
        # If the data is already batched and a new batch size is called,
        # then update the existing batch sizes. For unbatching the data,
        # update the batch state and then flatten the accessor array.
        if self._batch_size is not None:
            try:
                self._loader_accessors = np.concatenate(self._loader_accessors).ravel()
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
        num_splits = len(self._loader_accessors) // batch_size
        data_items = np.array(self._loader_accessors)
        overflow = len(self._loader_accessors) - num_splits * batch_size
        extra_items = data_items[-overflow:]
        try:
            batches = np.array_split(
                np.array(self._loader_accessors
                         [:num_splits * batch_size]), num_splits)
        except ValueError:
            log(f"There is less data ({len(self._loader_accessors)}) than the provided "
                f"batch size ({batch_size}). Consider using a smaller batch size.")
            batches = [self._loader_accessors]
        else:
            if len(extra_items) < batch_size:
                batches.append(extra_items)
        self._loader_accessors = np.array(batches, dtype = object)
        self._batch_size = batch_size

        # Update the batch creation method.
        self._make_batch = self._loaders[0]._manager._train_manager.make_batch

    def resize_images(self, image_size = None, method = 'bilinear'):
        """Resizes images within the loader to a specified size.

        This method applies a resizing parameter for images before they are
        returned from the data loader. The default starting point, if this
        method is never called, is to apply no resizing. If the loader is set
        in "training" mode and no size is specified, it defaults to (512, 512).

        Image resizing contains a few modes:

        1. `default` or `None`: No resizing, leaves images in default size.
           This is the default parameter if nothing is passed.
        2. `train`: This will set a default training size of (512, 512).
        3. `imagenet`: This will set a default size of (224, 224).
        4. custom size: Resizes the images to the provided size.

        The resizing also applies to the annotation in certain cases,
        depending on the task and the actual content of the annotation:

        - For object detection, the bounding box coordinates will
          be resized and the area of the box will in turn be recomputed.
        - For semantic segmentation, the annotation mask will be resized,
          using a nearest-neighbor interpolation to keep it as similar
          as possible to the original mask (preventing data loss).

        Parameters
        ----------
        image_size : optional
            The resizing parameter for the image.
        method : optional
            The method to resize the images. Should be one of 'nearest',
            'bilinear', 'linear', or 'cubic'. Defaults to 'bilinear'.

        Notes
        -----
        - There is no `auto` parameter for resizing images when using
          multiple data loaders. If auto is passed, it will warn the
          user and switch to `default`.
        - If a transform pipeline is provided, images will be resized
          *before* being passed into the transform pipeline.
        """
        if image_size == 'auto':
            log("There is no `auto` parameter for resizing images when using"
                "multiple datasets in a loader. Switching to `default`.")

        self._loaders.apply(
            lambda x: x.resize_images(
                image_size = image_size, method = method
            )
        )

    def transform(self,
                  transform = NoArgument,
                  target_transform = NoArgument,
                  dual_transform = NoArgument):
        """Applies vision transforms to the input image and annotation data.

        This method applies transformations to the image and annotation data
        in the dataset. Transforms include augmentations and other processing
        methods, and can be applied independently to the image and annotation,
        or together to both (`transform`, `target_transform`, `dual_transform`).

        The hierarchy in which transforms are applied is:

             transform  ->  --------|
                                    |----->   dual_transform
             target_transform ->  --|

        The `transform` and `target_transform` argument are used for methods
        which act independently on the image and the annotation, respectively.
        The values passed to these arguments can be:

            - An `albumentations` transform pipeline.
            - A `keras.Sequential` model (or preprocessing layer) or a
              set of `torchvision.transform`s.
            - A method which accepts one input and returns one output.

        The `dual_transform` argument is used for non-image-classification
        tasks. The following describe the types of arguments that can be
        passed to `dual_transform`, depending on the task:

        Object Detection:
            - An `albumentations` transform pipeline with `bbox_params` in
              to be applied to both the image and the bounding boxes.
            - A method (not a torchvision or Keras preprocessing pipeline)
              that accepts two inputs and returns two outputs.

        Semantic Segmentation:
            - An `albumentations` transform pipeline that may include
              spatial and/or visual augmentation.
            - A method to independently or dually apply transformations
              to the image and annotation mask.
            - A `torchvision.transforms` or `tf.keras.Sequential` pipeline
              which will be applied to the image and mask using the same
              random seed, for reproducibility. Use the provided method
              `generate_keras_segmentation_dual_transform` for this.

        If you want to reset the transforms, then simply call this method
        with no arguments. Alternatively, to reset just a single transform,
        pass the value of that argument as `None`.

        Parameters
        ----------
        transform : optional
            A transform to be applied independently to the input image.
        target_transform : optional
            A transform to be applied independently to the annotation.
        dual_transform : optional
            A transform to be applied to both the input and annotation.

        Notes
        -----
        - Image resizing takes place before any transformations are applied.
          After the transforms are applied in this order, they returned and
          if passed again, they will have a different transform applied to
          them. The state is independent of the images passed.
        - Albumentations transforms are special in that even transforms which
          would normally be passed to `dual_transform` (e.g., they act on the
          input image and the output annotation) can simply be passed to the
          `transform` argument and they will automatically be applied.
        """
        self._loaders.apply(
            lambda x: x._manager.push_transforms(
                transform = transform,
                target_transform = target_transform,
                dual_transform = dual_transform
            )
        )

    def normalize_images(self, method = 'scale'):
        """Converts images from 0-255 integers to 0-1 floats and normalizes.

        This is a convenience method to convert all images from integer-valued
        arrays into float-valued arrays, and normalizes them (using shifting
        and scaling from mean and std). This is useful for training in order
        to reduce computational complexity (instead of large-valued integer
        multiplication, only float multiplication), and for extracting the
        most information out of different types of imagery.

        There are three different 'normalization' modes that can be initialized
        with this method, as described below:

        1. `scale`: This simply scales images from the 0-255 pixel range to
           the 0-1 range (and converts them to floats as such).
        2. `imagenet`: This performs normalization using the traditional ImageNet
           mean and standard deviation:
                (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
           This is useful when trying to conduct transfer learning, for instance.
        3. `standard`: This performs normalization using a pre-calculated mean
           and standard deviation for the dataset (see the public sources JSON).

        To remove normalization altogether, pass `None` as a parameter.

        Parameters
        ----------
        method : str
            The method by which to normalize the images.

        Notes
        -----
        This method is not implicitly called when converting to PyTorch/TensorFlow
        mode, it needs to be manually called even if you just want 0-1 scaled images.
        """
        if method not in ['scale', 'imagenet', 'standard', None]:
            raise ValueError(f"Received invalid normalization method: '{method}'.")

        if method == 'scale':
            normalization_params = 'scale'
        elif method == 'imagenet':
            normalization_params = 'imagenet'
        elif method == 'standard':
            normalization_params = self._info.image_stats
        else:
            normalization_params = None

        # Normalization parameters may be specific to each dataset in
        # the loader, so we need to make sure we account for this.
        self._loaders.apply(
            lambda x, transform: x._manager.push_transforms(
                transform = transform,
                target_transform = NoArgument,
                dual_transform = NoArgument),
            args = [('normalize', normalization_params[key])
                    for key in self._loaders.keys]
        )

    def labels_to_one_hot(self):
        """Converts image classification numerical labels to one-hot labels.

        This is a convenience method to apply one-hot vector transformations
        to the output labels for image classification. Essentially, if we have
        a set of labels, [1, 2], it will convert it to [[0, 1, 0], [0, 0, 1]].
        This is a more commonly used format for image classification.
        """
        if self.task != 'image_classification':
            raise RuntimeError("The `one_hot` label transformation can only "
                               "be used for image classification tasks.")

        self.transform(
            target_transform = ('one_hot', self.num_classes)
        )

    def mask_to_channel_basis(self):
        """Converts semantic segmentation masks to channel-wise.

        This is a convenience method to convert integer-labeled semantic
        segmentation masks into channel-by-channel masks, essentially
        one-hot vector transformation but for semantic segmentation. Note
        that if the task is binary segmentation, e.g. there is only one
        class, then this method will do nothing.

        This method should traditionally be called *after* applying general
        transformations to the loader, in order to prevent any issues.
        """
        if self.task != 'semantic_segmentation':
            raise ValueError("The `mask_to_channel_basis` transformation "
                             "can only be used for semantic segmentation tasks.")

        self.transform(
            target_transform = ('channel_basis', self._info.num_classes)
        )

    def generalize_class_detections(self):
        """Generalizes object detection classes to a single class.

        This is a convenience method for object detection tasks, and
        converts all of the individual class labels in the task into
        a single class, essentially allowing the model to purely
        focus on detection of objects and fine-tuning bounding boxes,
        with no focus on differentiating classes of different boxes.
        """
        if self.task != 'object_detection':
            raise ValueError("The `generalize_class_detections` transformation"
                             "can only be used for object detection tasks.")

        self._loaders.apply(
            lambda x: x._manager._train_manager._set_annotation_remap_hook(
                AnnotationRemap(
                    self.class_to_num, self._info.num_to_class,
                    self.task, generalize_class_detections = True)))

    def export_contents(self, export_format = None):
        """Exports the internal contents of the `AgMLDataLoader`s.

        This method serves as a hook for high-level users who simply want
        to download and get the data, by exporting the unprocessed metadata
        of the actual dataset, with the following formats:

        Image Classification: A mapping between the local image paths and
            the numerical labels.
        Object Detection: A mapping between the local image paths (the full
            path, not just the file name), and the COCO JSON annotations
            corresponding to each of the images. To get the original COCO
            JSON annotation file contents, use `export_format = 'coco'`.
        Semantic Segmentation: A mapping between the local image paths
            and the local annotation mask paths.

        The `export_format` argument can be used to customize what this method
        returns. By default, it is set to `None`, and returns dictionaries with
        the above specified mappings. However, setting `export_format = 'arrays'`
        will return two arrays, with the first array containing the image paths
        and the second array containing the annotation data.

        Parameters
        ----------
        export_format : optional
            The format to export the data in. Defaults to a mapping.

        Returns
        -------
        The raw contents of the datasets.
        """
        return {
            k: self._loaders[k].export_contents(
                export_format = export_format)
            for k in self._loaders.keys
        }

    @staticmethod
    def _calculate_data_and_loader_index(index, bound_map, set_map):
        loader_idx = set_map[index]
        is_equal = np.where(index == np.array(list(bound_map.values())))[0]
        if is_equal.size != 0:
            data_idx = 0
        else:
            data_idx = int(index - list(bound_map.values())[int(
                np.searchsorted(np.array(
                    list(bound_map.values())), index) - 1)])
        if data_idx < 0:
            data_idx = 0
        return loader_idx, data_idx

    def _load_one_image_and_annotation(self, index):
        """Loads one image and annotation from a `DataObject`."""
        # Get the image and annotation from the corresponding loader.
        loader, data_idx = self._calculate_data_and_loader_index(
            index, self._bounds, self._set_to_keys)
        return self._loaders[loader][data_idx]

    def _load_multiple_items(self, indexes):
        """Loads multiple images and annotations from a set of `DataObject`s."""
        # Either we're getting multiple batches, or just multiple items.
        contents = []
        if self._batch_size is not None:
            for i in indexes:
                contents.append(self._load_batch(self._loader_accessors[i]))
        else:
            for i in indexes:
                contents.append(self._load_one_image_and_annotation(
                    self._loader_accessors[i]))
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
        if self.task == 'object_detection':
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
            image, annotation = self._load_one_image_and_annotation(index)
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
        return self._make_batch(
            images = images,
            annotations = annotations
        )

    def _get_item_impl(self, indexes):
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
                self._loader_accessors[indexes])

        # If we have a batch of images, then return the batch.
        if isinstance(indexes, int) and self._batch_size is not None:
            return self._load_batch(self._loader_accessors[indexes])

        # Otherwise, if there are multiple indexes (e.g., an unstacked
        # slice or just a tuple of integers), then we get multiple images.
        if isinstance(indexes, (list, tuple)):
            return self._load_multiple_items(indexes)

    def export_torch(self, **loader_kwargs):
        """Exports the contents of the loader in a native PyTorch loader.

        This method wraps the contents of this data loader inside of a
        `torch.utils.data.DataLoader`. This method differs from the
        `export_tensorflow()` method in that there is no need to convert
        directly to a `tf.data.Dataset`, rather if this `AgMLDataLoader`
        inherits from `torch.utils.data.Dataset`, it can just be directly
        wrapped into a `torch.utils.data.DataLoader`.

        The returned `DataLoader` is functionally similar to the
        `AgMLDataLoader` in terms of preprocessing and transforming. You
        can pass arguments to the `DataLoader` instantiation as keyword
        arguments to this method.

        Note that the `AgMLDataLoader` which this method encloses is
        instead a copy of the instance the method is run on, so that any
        changes to the loader afterwards don't affect the exported loader.

        Parameters
        ----------
        loader_kwargs : optional
            A set of keyword arguments for the `torch.utils.data.DataLoader`.
            See the documentation for the loader for more information.

        Returns
        -------
        A `torch.utils.data.DataLoader` enclosing a copy of this loader.
        """
        from agml.backend.tftorch import torch
        from torch.utils.data import DataLoader
        if get_backend() != 'torch':
            if user_changed_backend():
                raise StrictBackendError(
                    change = 'torch', obj = self.export_torch)
            set_backend('torch')

        # Make a copy of the `AgMLDataLoader` so the following changes
        # don't affect the original loader, just the new one.
        obj = self.copy()

        # Convert to a PyTorch dataset.
        obj.as_torch_dataset()

        # The `DataLoader` automatically batches objects using its
        # own mechanism, so we remove batching from this DataLoader.
        batch_size = loader_kwargs.pop(
            'batch_size', obj._batch_size)
        obj.batch(None)
        shuffle = loader_kwargs.pop(
            'shuffle', obj._shuffle_data)

        # The `collate_fn` for object detection is different because
        # the COCO JSON dictionaries each have different formats. So,
        # we need to replace it with a custom function.
        collate_fn = loader_kwargs.pop('collate_fn')
        if obj.task == 'object_detection' and collate_fn is None:
            def collate_fn(batch):
                images = torch.stack(
                    [i[0] for i in batch], dim = 0)
                coco = tuple([i[1] for i in batch])
                return images, coco

        # Return the DataLoader with a copy of this AgMLDataLoader, so
        # that changes to this will not affect the returned loader.
        return DataLoader(
            obj,
            batch_size = batch_size,
            shuffle = shuffle,
            collate_fn = collate_fn,
            **loader_kwargs
        )






