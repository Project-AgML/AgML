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

import copy
import fnmatch
import glob
import json
import os
from collections.abc import Sequence
from decimal import Decimal, getcontext
from typing import Union

import numpy as np

from agml.backend.config import SUPER_BASE_DIR, data_save_path, synthetic_data_save_path
from agml.backend.experimental import AgMLExperimentalFeatureWrapper
from agml.backend.tftorch import (
    StrictBackendError,
    _add_dataset_to_mro,  # noqa
    get_backend,
    set_backend,
    user_changed_backend,
)
from agml.data.builder import DataBuilder
from agml.data.exporters.yolo import export_yolo
from agml.data.manager import DataManager
from agml.data.metadata import DatasetMetadata, make_metadata
from agml.data.public import public_data_sources
from agml.framework import AgMLSerializable
from agml.utils.data import load_public_sources
from agml.utils.general import NoArgument, resolve_list_value
from agml.utils.io import get_dir_list, get_file_list
from agml.utils.logging import log
from agml.utils.random import inject_random_state
from agml.viz.general import show_sample


class AgMLDataLoaderMeta(type):
    def __instancecheck__(self, instance):
        # This override allows for objects of type `AgMLMultiDatasetLoader`
        # to be treated as an `AgMLDataLoader` when the following command
        # is run: `isinstance(a, AgMLDataLoader)` (hacky fix, essentially).
        if isinstance(instance, self.__class__):
            return True
        from agml.data.multi_loader import AgMLMultiDatasetLoader

        if isinstance(instance, AgMLMultiDatasetLoader):
            return True
        return False


class AgMLDataLoader(AgMLSerializable, metaclass=AgMLDataLoaderMeta):
    """Loads and provides a processing interface for a dataset.

    The `AgMLDataLoader` is the main interface to AgML's public dataset
    interface, and exposes an API which enables the downloading and
    subsequent local loading of a public dataset, as well as various
    preprocessing functions and hooks to integrate into existing pipelines.

    Methods provided include splitting the dataset into train/val/test sets,
    batching the data, applying transforms, and more. All of the processing
    code is contained internally, so all you need to do is instantiate the
    loader and call the relevant methods to apply the preprocessing methods.

    `AgMLDataLoader` supports both TensorFlow and PyTorch as backends, and
    can automatically perform tensor conversion and batching to enable
    seamless usage in training or inference pipelines. Data can also be
    exported into native TensorFlow and PyTorch objects.

    There is also support for using custom datasets outside of the AgML
    public data repository. To do this, you need to pass an extra argument
    containing metadata for the dataset, after which point the loader
    will work as normal (and all interfaces, except for the info parameters
    which are not provided, will also be available for standard use).

    Parameters
    ----------
    dataset : str
        The name of the public dataset you want to load. See the helper
        method `agml.data.public_data_sources()` for a list of datasets.
        If using a custom dataset, this can be any valid string.
    kwargs : dict, optional
        dataset_path : str, optional
            A custom path to download and load the dataset from.
        overwrite : bool, optional
            Whether to rewrite and re-install the dataset.
        meta : dict, optional
            A dictionary consisting of metadata properties, if you want
            to create a custom loader. At minimum, this needs to contain
            two parameters: `task`, indicating the type of machine learning
            task that the dataset is for, and `classes`, a list of the
            classes that the dataset contains.

    Notes
    -----
    See the methods for examples on how to use an `AgMLDataLoader` effectively.
    """

    IS_MULTI_DATASET: bool = False

    serializable = frozenset(
        (
            "info",
            "builder",
            "manager",
            "train_data",
            "train_content",
            "val_data",
            "val_content",
            "test_data",
            "test_content",
            "is_split",
            "meta_properties",
        )
    )

    def __new__(cls, dataset, **kwargs):
        # If a single dataset is passed, then we use the base `AgMLDataLoader`.
        # However, if an iterable of datasets is passed, then we need to
        # dispatch to the subclass `AgMLMultiDatasetLoader` for them.
        if isinstance(dataset, (str, DatasetMetadata)):
            if "*" in dataset:  # enables wildcard search for datasets
                valid_datasets = fnmatch.filter(load_public_sources().keys(), dataset)
                if len(valid_datasets) == 0:
                    raise ValueError(f"Wildcard search for dataset '{dataset}' yielded no results.")
                if len(valid_datasets) == 1:
                    log(
                        f"Wildcard search for dataset '{dataset}' yielded only "
                        f"one result. Returning a regular, single-element data loader."
                    )
                    return super(AgMLDataLoader, cls).__new__(cls)
                from agml.data.multi_loader import AgMLMultiDatasetLoader

                return AgMLMultiDatasetLoader(valid_datasets, **kwargs)
            return super(AgMLDataLoader, cls).__new__(cls)
        elif isinstance(dataset, Sequence):
            if len(dataset) == 1:
                log(
                    "Received a sequence with only one element when "
                    "instantiating an `AgMLDataLoader`. Returning "
                    "a regular, single-element data loader."
                )
                return super(AgMLDataLoader, cls).__new__(cls)
            from agml.data.multi_loader import AgMLMultiDatasetLoader

            return AgMLMultiDatasetLoader(dataset, **kwargs)
        raise TypeError(
            f"Expected either a single dataset name (or metadata), or"
            f"a list of dataset names/metadata when instantiating an "
            f"`AgMLDataLoader`. Got {dataset} of type {type(dataset)}."
        )

    def __getnewargs__(self):
        return (self._info.name,)

    def __init__(self, dataset, **kwargs):
        """Instantiates an `AgMLDataLoader` with the dataset."""
        # Set up the dataset and its associated metadata.
        self._info = make_metadata(dataset, kwargs.get("meta", None))

        # The data for the class is constructed in two stages. First, the
        # internal contents are constructed using a `DataBuilder`, which
        # finds and wraps the local data in a proper format.
        self._builder = DataBuilder(
            info=self._info,
            dataset_path=kwargs.get("dataset_path", None),
            overwrite=kwargs.get("overwrite", False),
        )

        # These contents are then passed to a `DataManager`, which conducts
        # the actual loading and processing of the data when called.
        self._manager = DataManager(
            builder=self._builder,
            task=self._info.tasks.ml,
            name=self._info.name,
            root=self._builder.dataset_root,
        )

        # If the dataset is split, then the `AgMLDataLoader`s with the
        # split and reduced data are stored as accessible class properties.
        self._train_data = None
        self._train_content = None
        self._val_data = None
        self._val_content = None
        self._test_data = None
        self._test_content = None
        self._is_split = False

        # Set the direct access metadata properties like `num_images` and
        # `classes`, since these can be modified depending on the state of
        # the loader, whilst the `info` parameter attributes cannot.
        self._meta_properties = {
            "num_images": self._info.num_images,
            "classes": self._info.classes,
            "num_classes": self._info.num_classes,
            "num_to_class": self._info.num_to_class,
            "class_to_num": self._info.class_to_num,
            "data_distributions": {self.name: self._info.num_images},
        }

    @classmethod
    def custom(cls, name, dataset_path=None, classes=None, **kwargs):
        """Creates an `AgMLDataLoader` with a set of custom data.

        If you have a custom dataset that you want to use in an `AgMLDataLoader`,
        this method constructs the loader using similar semantics to the regular
        loader instantiation. It is a wrapper around using the `meta` argument to
        provide dataset properties that provides additional convenience for some
        circumstances, as summarized below.

        Functionally, this method is equivalent to instantiating `AgMLDataLoader`
        with an extra argument `meta` that contains metadata for the dataset, with
        the `task` and `classes` keys required and the others not necessary. This
        would look like follows:

        > loader = AgMLDataLoader('name', meta = {'task': task, 'classes': classes})

        This method replaces the meta dictionary with keyword arguments to allow
        for a more Pythonic construction of the custom loader. This method, however
        includes additional optimizations which allow for a more convenient way
        to instantiate the loader:

        1. It automatically inferences the task from the structure which the data is
           in, so you don't need to provide the task at all to this method.
        2. For image classification and object detection task, this method will
           attempt to automatically inference the classes in the loader (by searching
           for the image directories for image classification tasks, and searching
           in the COCO JSON file for object detection). Semantic segmentation tasks,
           however, still require the list of classes to be passed.

        This makes it so that in a variety of cases, the loader can be instantiated
        without even requiring any metadata, as most of it can be inferred directly
        by this method and thus streamlines the procedure for using custom data.

        If you want to cache the metadata, rather than constantly putting them as
        arguments, then create a file `.meta.json` at the path `/root/.meta.json`
        with the parameters that you want.

        Parameters
        ----------
        name : str
            A name for the custom dataset (this can be any valid string). This
            can also be a path to the dataset (in which case the name will be
            the base directory inferred from the path).
        dataset_path : str, optional
            A custom path to load the dataset from. If this is not passed,
            we will assume that the dataset is at the traditional path:
            `~/.agml/datasets/<name>` (or the changed default data path).
            Otherwise, the dataset can be passed as a path such as `/root/name`,
            or `/root`; in the latter case the method will check for `/root/name`.
        classes : list, tuple
            A list of string-labels for the classes of the dataset, in order.
            This is not required for image classification/object detection.
        kwargs : dict
            Any other metadata for the dataset, this is not required.

        Returns
        -------
        An `AgMLDataLoader` outfitted with the custom dataset.
        """
        # Check the name and ensure that no dataset with that name exists.
        if name in load_public_sources().keys() or not isinstance(name, str):
            raise ValueError(
                f"Invalid name '{name}', the name should be "
                f"a string that is not an existing dataset in "
                f"the AgML public data source repository."
            )

        # Check if the `name` is itself the path to the dataset.
        if os.path.exists(name):
            dataset_path = name
            name = os.path.basename(name)

        # Locate the path to the dataset.
        if dataset_path is None:
            dataset_path = os.path.abspath(os.path.join(data_save_path(), name))
            if not os.path.exists(dataset_path):
                raise NotADirectoryError(
                    f"Existing directory '{dataset_path}' for dataset of name "
                    f"{name} not found, pass a custom path if you want to use "
                    f"a custom dataset path for the dataset."
                )
        else:
            dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
            if not os.path.exists(dataset_path):
                if not os.path.exists(dataset_path):
                    raise NotADirectoryError(
                        f"Could not find a directory for dataset '{name}' at the "
                        f"provided dataset path: {dataset_path}."
                    )
            if not dataset_path.endswith(name):
                dataset_path = os.path.join(dataset_path, name)
                if not os.path.exists(dataset_path):
                    raise NotADirectoryError(
                        f"Could not find a directory for dataset '{name}' at the "
                        f"provided dataset path: {dataset_path}."
                    )

        # Infer the task based on the provided dataset path.
        if os.path.exists(os.path.join(dataset_path, "annotations.json")):
            task = "object_detection"
        elif os.path.exists(os.path.join(dataset_path, "images")) and os.path.exists(
            os.path.join(dataset_path, "annotations")
        ):
            task = "semantic_segmentation"
        elif len(get_file_list(dataset_path)) == 0 and len(get_dir_list(dataset_path)) != 0:
            task = "image_classification"
        else:
            raise TypeError("Unrecognized dataset annotation format.")

        # Check if there is a metadata file.
        kwargs["classes"] = classes
        if os.path.exists(os.path.join(dataset_path, ".meta.json")):
            with open(os.path.join(dataset_path, ".meta.json"), "r") as f:
                kwargs.update(json.load(f))

        # Infer the classes for image classification/object detection.
        classes = kwargs.pop("classes")
        if classes is None:
            if task == "semantic_segmentation":
                raise ValueError("Classes are required for a semantic segmentation task.")
            elif task == "image_classification":
                classes = get_dir_list(dataset_path)
            else:  # object detection
                with open(os.path.join(dataset_path, "annotations.json"), "r") as f:
                    classes = [c["name"] for c in json.load(f)["categories"]]

        # Construct and return the `AgMLDataLoader`.
        return cls(
            name,
            dataset_path=dataset_path,
            meta={"task": task, "classes": classes, **kwargs},
        )

    @classmethod
    def helios(cls, name, dataset_path=None):
        """Creates an `AgMLDataLoader` from a Helios-generated dataset.
        Given the path to a Helios-generated (and converted) dataset, this method
        will generate an `AgMLDataLoader` which is constructed using similar
        semantics to the regular instantiation. This method is largely similar to
        `AgMLDataLoader.custom()`, but also takes into account the extra
        information which is provided in the `.metadata` directory of the Helios
        generated dataset, allowing it to contain potentially even more info.
        """
        # Instantiate from a list of datasets.
        if isinstance(name, (list, tuple)):
            if dataset_path is None:
                dataset_path = [None] * len(name)
            elif isinstance(dataset_path, str):
                dataset_path = [dataset_path] * len(name)
            else:
                if not len(dataset_path) == len(name):
                    raise ValueError("The number of dataset paths must be " "the same as the number of dataset names.")
            datasets = [cls.helios(n, dataset_path=dp) for n, dp in zip(name, dataset_path)]
            return cls.merge(*datasets)

        # Instantiate from a wildcard pattern.
        if isinstance(name, str) and "*" in name:
            if dataset_path is None:
                dataset_path = os.path.abspath(synthetic_data_save_path())
            elif not os.path.exists(dataset_path):
                raise NotADirectoryError(
                    f"Existing directory '{dataset_path}' for dataset of name "
                    f"{name} not found, pass a custom path if you want to use "
                    f"a custom dataset path for the dataset."
                )

            # Get the list of datasets.
            possible_datasets = glob.glob(os.path.join(dataset_path, name))
            if len(possible_datasets) == 0:
                raise ValueError(f"No datasets found for pattern: {name}.")
            datasets = [cls.helios(os.path.basename(p), dataset_path=dataset_path) for p in sorted(possible_datasets)]
            return cls.merge(*datasets)

        # Locate the path to the dataset, using synthetic semantics.
        if dataset_path is None:
            dataset_path = os.path.abspath(os.path.join(synthetic_data_save_path(), name))
            if not os.path.exists(dataset_path):
                raise NotADirectoryError(
                    f"Existing directory '{dataset_path}' for dataset of name "
                    f"{name} not found, pass a custom path if you want to use "
                    f"a custom dataset path for the dataset."
                )
        else:
            dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
            if not os.path.exists(dataset_path):
                if not os.path.exists(dataset_path):
                    raise NotADirectoryError(
                        f"Could not find a directory for Helios dataset '{name}' "
                        f"at the provided dataset path: {dataset_path}."
                    )

            # just in case there is a locally defined folder with the same name
            # as a dataset in the `~/.agml/synthetic` directory, warn in advance:
            if os.path.exists(os.path.join(os.path.abspath(synthetic_data_save_path()), name)):
                log(
                    f"Found a dataset folder '{name}' in the synthetic data "
                    f"directory, which may conflict with the Helios dataset."
                )
            if not dataset_path.endswith(name):
                dataset_path = os.path.join(dataset_path, name)
                if not os.path.exists(dataset_path):
                    raise NotADirectoryError(
                        f"Could not find a directory for Helios dataset '{name}' "
                        f"at the provided dataset path: {dataset_path}."
                    )

        # Load the information file.
        info_file = os.path.join(dataset_path, ".metadata", "agml_info.json")
        if not os.path.exists(info_file):
            raise FileNotFoundError(
                f"The information file at '{info_file}' for the " f"Helios dataset {name} could not be found."
            )
        with open(info_file, "r") as f:
            meta = json.load(f)

        # Construct the loader.
        return cls.custom(name, dataset_path, **meta)
    
    @classmethod
    def from_parent(cls, parent_dataset, filters=None, **kwargs):
        """Instantiates an `AgMLDataLoader` from a parent dataset.
        
        Given a selected `parent_dataset`, that is, a larger super-dataset which
        contains multiple sub-datasets, this method will construct an `AgMLDataLoader`
        containing all of the datasets within the parent dataset (or a specific
        subset of datasets from the parent, depending on the keyword arguments).
        
        As a defining example, AgML provides access to the iNatAg dataset, which is
        composed as a collection of many individual species sub-datasets. Here, iNatAg
        and iNatAg-mini are the parent datasets, and the individual species datasets
        are the sub-datasets. To load the entirety of either of these datasets, you
        would use this method with the parent dataset name as the argument to load
        the entire dataset into a singular loader.

        **Note**: Filtering functionality is still to be implemented. At the moment,
        you can either load the entire dataset or need to manually select which
        subsets you want; this will be augmented in the future.

        Parameters
        ----------
        parent_dataset : str
            The name of the parent dataset to load.
        """
        if not isinstance(filters, dict):
            raise ValueError("You should provide a dictionary of filters with the names "
                             "and values of the various filters which you desire to load.")
        
        # Get all of the subdatasets from the parent dataset, and construct the loader.
        subdataset_sources = public_data_sources(parent_dataset=parent_dataset)

        if filters is not None:
            valid_subdataset_sources = []
            for source in subdataset_sources:
                for key, value in filters.items():
                    if isinstance(source.extra_metadata[key], list):
                        if isinstance(value, list):
                            if any(v in source.extra_metadata[key] for v in value):
                                valid_subdataset_sources.append(source)
                        elif value in source.extra_metadata[key]:
                            valid_subdataset_sources.append(source)
                    elif isinstance(value, list):
                        if source.extra_metadata[key] in value:
                            valid_subdataset_sources.append(source)
                    elif source.extra_metadata[key] == value:
                        valid_subdataset_sources.append(source)
            subdataset_sources = valid_subdataset_sources

            if len(subdataset_sources) == 0:
                raise ValueError(f"No datasets in {parent_dataset} found matching the provided filters.")

        subdatasets = [source['name'] for source in subdataset_sources]
        return cls(subdatasets, **kwargs, parent_dataset = parent_dataset, parent_dataset_filters = filters)

    @staticmethod
    def merge(*loaders, classes=None):
        """Merges a set of `AgMLDataLoader`s into a single loader.

        Given a set of input `AgMLDataLoader`s, this method will return a single
        `AgMLDataLoader` which is capable of returning data from any and every one
        of the input loaders. The resultant loader is functionally equivalent to
        the `AgMLDataLoader` returned by instantiating an `AgMLDataLoader` from a
        sequence of AgML public data sources, except that in this case, the input
        loaders may be subject to a number of input modifications before merging.

        This also allows the usage of both an AgML public data source and a custom
        dataset together in a single multi-dataset loader. As such, this method
        should be used with caution, as since input loaders may be allowed to have
        any modification, certain methods may not function as expected. For instance,
        if one of the passed loaders has already been split, then the overall new
        multi-loader cannot be split as a whole. Similarly, if also using a custom
        dataset, then any properties of the `info` parameter which are not passed
        to the dataset cannot be used, even if the other datasets have them.

        Parameters
        ----------
        loaders : Tuple[AgMLDataLoader]
            A collection of `AgMLDataLoader`s (but not any `AgMLDataLoader`s
            which are already holding a collection of datasets).
        classes : list
            A list of classes in the new loader. This argument can be used to
            construct a custom ordering (non-alphabetical) of classes in the loader.

        Returns
        -------
        A new `AgMLDataLoader` wrapping the input datasets.
        """
        # Validate the input loaders.
        from agml.data.multi_loader import AgMLMultiDatasetLoader

        if len(loaders) == 1:
            raise ValueError("There should be at least two inputs to the `merge` method.")
        for loader in loaders:
            if isinstance(loader, AgMLMultiDatasetLoader):
                raise TypeError("Cannot merge datasets which already hold a " "collection of multiple datasets.")

        # Instantiate the `AgMLMultiDatasetLoader`.
        return AgMLMultiDatasetLoader._instantiate_from_collection(*loaders, classes=classes)

    def __add__(self, other):
        if not isinstance(other, AgMLDataLoader):
            return NotImplemented
        return AgMLDataLoader.merge(self, other)

    def __len__(self):
        return self._manager.data_length()

    def __getitem__(self, indexes: Union[list, int, slice]):
        if isinstance(indexes, slice):
            data = np.arange(self._manager.data_length())
            indexes = data[indexes].tolist()
        if isinstance(indexes, int):
            indexes = [indexes]
        if np.isscalar(indexes):
            indexes = [indexes.item()]  # noqa
        for idx in indexes:
            if idx not in range(len(self)):
                if idx not in [-i for i in range(1, len(self) + 1, 1)]:
                    raise IndexError(f"Index {idx} out of range of " f"AgMLDataLoader length: {len(self)}.")
        return self._manager.get(resolve_list_value(indexes))

    def __iter__(self):
        for indx in range(len(self)):
            yield self[indx]

    def __repr__(self):
        out = f"<AgMLDataLoader: (dataset={self.name}"
        out += f", task={self.task}"
        out += f", images={self.num_images}"
        out += f") at {hex(id(self))}>"
        return out

    def __str__(self):
        return repr(self)

    def __copy__(self):
        """Copies the loader and updates its state."""
        cp = super(AgMLDataLoader, self).__copy__()
        cp.copy_state(self)
        return cp

    def copy(self):
        """Returns a deep copy of the data loader's contents."""
        return self.__copy__()

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
        # Re-construct the training manager.
        new_train_manager = loader._manager._train_manager.__copy__()
        self._manager._train_manager = new_train_manager

        # Re-construct the transform manager.
        new_transform_manager = loader._manager._transform_manager.__copy__()
        self._manager._transform_manager = new_transform_manager
        self._manager._train_manager._transform_manager = new_transform_manager

        # Re-construct the resizing manager.
        new_resize_manager = loader._manager._resize_manager.__copy__()
        self._manager._resize_manager = new_resize_manager
        self._manager._train_manager._resize_manager = new_resize_manager

    @property
    def name(self):
        """Returns the name of the dataset in the loader."""
        return self._info.name

    @property
    def dataset_root(self):
        """Returns the local path to the dataset being used."""
        return self._builder.dataset_root

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
        return self._info.tasks.ml

    @property
    def num_images(self):
        """Returns the number of images in the dataset."""
        return self._meta_properties.get("num_images")

    @property
    def classes(self):
        """Returns the classes that the dataset is predicting."""
        return self._meta_properties.get("classes")

    @property
    def num_classes(self):
        """Returns the number of classes in the dataset."""
        return self._meta_properties.get("num_classes")

    @property
    def num_to_class(self):
        """Returns a mapping from a number to a class label."""
        return self._meta_properties.get("num_to_class")

    @property
    def class_to_num(self):
        """Returns a mapping from a class label to a number."""
        return self._meta_properties.get("class_to_num")

    @property
    def data_distributions(self):
        """Displays the distribution of images from each source."""
        return self._meta_properties.get("data_distributions")

    @property
    def image_size(self):
        """Returns the determined image size for the loader.

        This is primarily useful when using auto shape inferencing, to
        access what the final result ends up being. Otherwise, it may
        just return `None` or the shape that the user has set.
        """
        return self._manager._resize_manager.size

    def _generate_split_loader(self, contents, split, meta_properties=None, **kwargs):
        """Generates a split `AgMLDataLoader`."""
        # Check if the data split exists.
        if contents is None:
            raise ValueError(f"Attempted to access '{split}' split when " f"the data has not been split for '{split}'.")

        # Load a new `DataManager` and update its internal managers
        # using the state of the existing loader's `DataManager`.
        builder = DataBuilder.from_data(
            contents=[contents, kwargs.get("labels_for_image", None)],
            info=self.info,
            root=self.dataset_root,
            builder=self._builder,
        )
        current_manager = copy.deepcopy(self._manager.__getstate__())
        current_manager.pop("builder")
        current_manager["builder"] = builder

        # Build the new accessors and construct the `DataManager`.
        accessors = np.arange(len(builder.get_contents()))
        if self._manager._shuffle:
            np.random.shuffle(accessors)
        current_manager["accessors"] = accessors
        batch_size = current_manager.pop("batch_size")
        current_manager["batch_size"] = None
        new_manager = DataManager.__new__(DataManager)
        new_manager.__setstate__(current_manager)

        # After the builder and accessors have been generated, we need
        # to generate a new list of `DataObject`s.
        new_manager._create_objects(new_manager._builder, self.task)

        # Update the `TransformManager` and `ResizeManager` of the
        # `TrainManager` in the `DataManager` (they need to be synchronized).
        new_manager._train_manager._transform_manager = new_manager._transform_manager
        new_manager._train_manager._resize_manager = new_manager._resize_manager

        # Batching data needs to be done independently.
        if batch_size is not None:
            new_manager.batch_data(batch_size=batch_size)

        # Update the metadata parameters.
        if meta_properties is None:
            meta_properties = self._meta_properties.copy()
            meta_properties["num_images"] = len(contents)
            meta_properties["data_distributions"] = {self.name: len(contents)}

        # Instantiate a new `AgMLDataLoader` from the contents.
        loader_state = self.copy().__getstate__()
        loader_state["builder"] = builder
        loader_state["manager"] = new_manager
        loader_state["meta_properties"] = meta_properties
        cls = super(AgMLDataLoader, self).__new__(AgMLDataLoader)
        cls.__setstate__(loader_state)
        for attr in ["train", "val", "test"]:
            setattr(cls, f"_{attr}_data", None)
        cls._is_split = True
        return cls

    @property
    def train_data(self):
        """Stores the `train` split of the data in the loader."""
        if isinstance(self._train_data, AgMLDataLoader):
            return self._train_data
        self._train_data = self._generate_split_loader(self._train_content, split="train")
        return self._train_data

    @property
    def val_data(self):
        """Stores the `val` split of the data in the loader."""
        if isinstance(self._val_data, AgMLDataLoader):
            return self._val_data
        self._val_data = self._generate_split_loader(self._val_content, split="val")
        self._val_data.eval()
        return self._val_data

    @property
    def test_data(self):
        """Stores the `test` split of the data in the loader."""
        if isinstance(self._test_data, AgMLDataLoader):
            return self._test_data
        self._test_data = self._generate_split_loader(self._test_content, split="test")
        self._test_data.eval()
        return self._test_data

    def eval(self) -> "AgMLDataLoader":
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
        self._manager.update_train_state("eval")
        return self

    def disable_preprocessing(self) -> "AgMLDataLoader":
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
        self._manager.update_train_state(False)
        return self

    def reset_preprocessing(self) -> "AgMLDataLoader":
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
        self._manager.update_train_state(None)
        return self

    def on_epoch_end(self):
        """Shuffles the dataset on the end of an epoch for a Keras sequence.

        If `as_keras_sequence()` is called and the `AgMLDataLoader` inherits
        from `tf.keras.utils.Sequence`, then this method will shuffle the
        dataset on the end of each epoch to improve training.
        """
        self._manager._maybe_shuffle()

    def as_keras_sequence(self) -> "AgMLDataLoader":
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

        _add_dataset_to_mro(self, "tf")
        self._manager.update_train_state("tf")
        return self

    def as_torch_dataset(self) -> "AgMLDataLoader":
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
        _add_dataset_to_mro(self, "torch")
        self._manager.update_train_state("torch")
        return self

    @property
    def shuffle_data(self):
        """Returns whether the loader is set to shuffle data or not.

        By default, if no value is passed in initialization, this is set to
        `True`. It can be manually toggled to `False` using this property.
        """
        return self._manager._shuffle

    @shuffle_data.setter
    def shuffle_data(self, value):
        """Set whether the loader should shuffle data or not.

        This can be used to enable/disable shuffling, by passing
        either `True` or `False`, respectively.
        """
        if not isinstance(value, bool):
            raise TypeError("Expected either `True` or `False` for 'shuffle_data'.")
        self._manager._shuffle = value

    def shuffle(self, seed=None):
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
        self._manager.shuffle(seed=seed)
        return self

    def take_images(self):
        """Returns a mini-loader over all of the images in the dataset.

        This method returns a mini-loader over all of the images in the dataset,
        without any annotations. This is useful for running inference over just
        the images in a dataset, or in general any operations in which you just
        want the raw image data from a loader, without any corresponding labels.

        Returns
        -------
        An `agml.data.ImageLoader` with the dataset images.
        """
        from agml.data.image_loader import ImageLoader

        return ImageLoader(self)

    def take_dataset(self, name) -> "AgMLDataLoader":
        """Takes one of the datasets in a multi-dataset loader.

        This method selects one of the datasets (as denoted by `name`)
        in this multi-dataset collection and returns an `AgMLDataLoader`
        with its contents. These contents will be subject to any transforms
        and modifications as applied by the main loader, but the returned
        loader will be a copy, such that any new changes made to the main
        multi-dataset loader will not affect the new loader.

        Note that this method only works for multi-dataset collections.

        Parameters
        ----------
        name : str
            The name of one of the sub-datasets of the loader.

        Returns
        -------
        An `AgMLDataLoader`.
        """
        raise ValueError("The `loader.take_dataset` method only works for multi-dataset loaders.")

    def take_class(self, classes, reindex=True) -> "AgMLDataLoader":
        """Reduces the dataset to a subset of class labels.

        This method, given a set of either integer or string class labels,
        will return a new `AgMLDataLoader` containing a subset of the
        original dataset, where the only classes in the dataset are those
        specified in the `classes` argument.

        The new loader will have info parameters like `num_classes` and
        `class_to_num` updated for the new set of classes; however, the
        original `info` metadata will remain the same as the original.

        Note that if the dataset contains images which have bounding boxes
        corresponding to multiple classes, this method will not work.

        Parameters
        ----------
        classes : list, int, str
            Either a single integer/string for a single class, or a list
            of integers or strings for multiple classes. Integers should
            be one-indexed for object detection.
        reindex : bool
            Re-indexes all of the new classes starting from 1, in ascending
            order based on their number in the original dataset.

        Notes
        -----
        This method only works for object detection datasets.
        """
        if self._info.tasks.ml != "object_detection":
            raise RuntimeError("The `take_class` method can only be " "used for object detection datasets.")

        # Parse the provided classes and determine their numerical labels.
        if isinstance(classes, str):
            if classes not in self.classes:
                raise ValueError(
                    f"Received a class '{classes}' for `loader.take_class`, "
                    f"which is not in the classes for {self.name}: {self.classes}"
                )
            classes = [self.class_to_num[classes]]
        elif isinstance(classes, int):
            try:
                self.num_to_class[classes]
            except IndexError:
                raise ValueError(
                    f"The provided class number {classes} is out of "
                    f"range for {self.num_classes} classes. Make sure "
                    f"you are using zero-indexing."
                )
            classes = [classes]
        else:
            parsed_classes = []
            if isinstance(classes[0], str):
                for cls in classes:
                    if cls not in self.classes:
                        raise ValueError(
                            f"Received a class '{cls}' for `loader.take_class`, which "
                            f"is not in the classes for {self.name}: {self.classes}"
                        )
                    parsed_classes.append(self.class_to_num[cls])
            elif isinstance(classes[0], int):
                for cls in classes:
                    try:
                        self.num_to_class[cls]
                    except IndexError:
                        raise ValueError(
                            f"The provided class number {cls} is out of "
                            f"range for {self.num_classes} classes. Make "
                            f"sure you are using zero-indexing."
                        )
                    parsed_classes.append(cls)
            classes = parsed_classes.copy()

        # Ensure that there are no images with multi-category boxes.
        categories = self._builder._labels_for_image
        if not all(len(np.unique(c)) == 1 for c in categories.values()):
            raise ValueError(
                f"Dataset {self.name} has images with multiple categories for "
                f"bounding boxes, cannot take an individual set of classes."
            )

        # Get the new data which will go in the loader. The `DataBuilder`
        # stores a mapping of category IDs corresponding to the bounding
        # boxes in each image, so we use these to determine the new boxes.
        new_category_map = {k: v for k, v in categories.items() if v[0] in classes}
        new_coco_map = {k: v for k, v in self._builder._data.items() if k in new_category_map.keys()}

        # Create the new info parameters for the class. If reindexing
        # is requested, then we re-index the classes based on the order
        # in which they are given, and then create a new dictionary
        # to map the original annotations to the new ones (used later).
        if reindex:
            old_to_new = {cls: idx + 1 for idx, cls in enumerate(classes)}
            new_classes = [self.num_to_class[c] for c in classes]
            new_properties = {
                "num_images": len(new_coco_map.keys()),
                "classes": new_classes,
                "num_classes": len(new_classes),
                "num_to_class": {i + 1: c for i, c in enumerate(new_classes)},
                "class_to_num": {c: i + 1 for i, c in enumerate(new_classes)},
            }
        else:
            new_classes = [self.num_to_class[c] for c in classes]
            new_properties = {
                "num_images": len(new_coco_map.keys()),
                "classes": new_classes,
                "num_classes": len(classes),
                "num_to_class": {c: self.num_to_class[c] for c in classes},
                "class_to_num": {self.num_to_class[c]: c for c in classes},
            }

        # Create the new loader.
        obj = self._generate_split_loader(
            new_coco_map,
            "train",
            meta_properties=new_properties,
            labels_for_image=new_category_map,
        )
        obj._is_split = False

        # Re-index the loader if requested to.
        if reindex:

            class AnnotationRemap(AgMLSerializable):
                """A helper class to remap annotation labels for multiple datasets."""

                serializable = frozenset(("map",))

                def __init__(self, o2n):
                    self._map = o2n

                def __call__(self, contents, name):
                    """Re-maps the annotation for the new, multi-dataset mapping."""
                    image, annotations = contents

                    # Re-map the annotation ID.
                    category_ids = annotations["category_id"]
                    category_ids[np.where(category_ids == 0)[0]] = 1  # fix
                    new_ids = np.array([self._map[c] for c in category_ids])
                    annotations["category_id"] = new_ids
                    return image, annotations

            # Maps the annotations.
            obj._manager._train_manager._set_annotation_remap_hook(AnnotationRemap(old_to_new))  # noqa

        # Return the loader.
        return obj

    @inject_random_state
    def take_random(self, k) -> "AgMLDataLoader":
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

        Parameters
        ----------
        k : int, float
            Either an integer specifying the number of samples or a float
            specifying the proportion of images from the total to take.
        {random_state}

        Returns
        -------
        A reduced `AgMLDataLoader` with the new data.
        """
        # Parse the input to an integer.
        if isinstance(k, float):
            # Check that 0.0 <= k <= 1.0.
            if not 0.0 <= k <= 1.0:
                raise ValueError("If passing a proportion to `take_class`, " "it should be in range [0.0, 1.0].")

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
                    f"{k} images, when the dataset has {self.num_images}."
                )

            # We use a similar functionality to the `split` method here,
            # essentially choosing a random sampling up until `k` and then
            # using the `DataManager` to access the reduced data.
            split = np.arange(0, self.num_images)
            np.random.shuffle(split)
            indices = split[:k]
            content = list(self._manager.generate_split_contents({"content": indices}).values())[0]

            # Create a new `AgMLDataLoader` from the new contents.
            obj = self._generate_split_loader(content, "train")
            obj._is_split = False
            return obj

        # Otherwise, raise an error.
        else:
            raise TypeError(f"Expected only an int or a float when " f"taking a random split, got {type(k)}.")

    @inject_random_state
    def split(self, train=None, val=None, test=None, shuffle=True):
        """Splits the data into train, val and test splits.

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
        train : int, float
            The split for training data.
        val : int, float
            The split for validation data.
        test : int, float
            The split for testing data.
        shuffle : bool
            Whether to shuffle the split data.
        {random_state}

        Notes
        -----
        Any processing applied to this `AgMLDataLoader` will also be present
        in the split loaders until they are accessed from the class. If you
        don't want these to be applied, access them right after splitting.
        """
        # Check if the data is already split or batched.
        if not AgMLExperimentalFeatureWrapper.nested_splitting():
            if self._is_split:
                raise ValueError("Cannot split already split data.")
        elif self._manager._batch_size is not None:
            raise ValueError("Cannot split already batched data. " "Split the data before batching.")

        # If no parameters are passed, then don't do anything.
        arg_dict = {"train": train, "val": val, "test": test}
        valid_args = {k: v for k, v in arg_dict.items() if v is not None}
        if all(i is None for i in arg_dict.values()):
            return None

        # There are two valid ways to pass splits. The first involves
        # passing the split values as floats, the second as ints. If we
        # receive the splits as floats, then we convert them to ints
        # in order to maintain maximum precision if we do manage to get
        # them as ints. Then the procedure is the same.
        if all(isinstance(i, float) for i in valid_args.values()):
            # To prevent potential precision errors, we need to convert the
            # splits to `Decimal` objects and then set the decimal precision.
            getcontext().prec = 4  # noqa
            valid_args = {k: Decimal(v) / Decimal(1) for k, v in valid_args.items()}
            if not sum(valid_args.values()) == Decimal(1):
                raise ValueError(
                    f"Got floats for input splits and expected a sum " f"of 1, instead got {sum(valid_args.values())}."
                )

            # Convert the splits from floats to ints. If the sum of the int
            # splits are greater than the total number of data, then the largest
            # split is decreased in order to keep compatibility in usage.
            num_images = self.num_images
            proportions = {k: int(v * Decimal(num_images)) for k, v in valid_args.items()}
            if sum(proportions.values()) != num_images:
                diff = sum(proportions.values()) - num_images
                largest_split = list(proportions.keys())[list(proportions.values()).index(max(proportions.values()))]
                proportions[largest_split] = proportions[largest_split] - diff
            valid_args = proportions.copy()

        # Create the actual data splits.
        if all(isinstance(i, int) for i in valid_args.values()):
            # Ensure that the sum of the splits is the length of the dataset.
            if not sum(valid_args.values()) == self.num_images:
                raise ValueError(
                    f"Got ints for input splits and expected a sum "
                    f"equal to the dataset length, {self.num_images},"
                    f"but instead got {sum(valid_args.values())}."
                )

            # The splits will be generated as sequences of indices.
            generated_splits = {}
            split = np.arange(0, self.num_images)
            names, splits = list(valid_args.keys()), list(valid_args.values())

            # Shuffling of the indexes will occur first, such that there is an even
            # shuffle across the whole sample and not just in the individual splits.
            if shuffle:
                np.random.shuffle(split)

            # Create the list of split indexes.
            if len(valid_args) == 1:
                generated_splits[names[0]] = split
            elif len(valid_args) == 2:
                generated_splits = {k: v for k, v in zip(names, [split[: splits[0]], split[splits[0] :]])}
            else:
                split_1 = split[: splits[0]]
                split_2 = split[splits[0] : splits[0] + splits[1]]
                split_3 = split[splits[0] + splits[1] :]
                generated_splits = {k: v for k, v in zip(names, [split_1, split_2, split_3])}

            # Get the actual split contents from the manager. These contents are
            # not `DataObject`s, rather they are simply the actual mapping of
            # data to be passed to a `DataBuilder` when constructing the splits.
            contents = self._manager.generate_split_contents(generated_splits)

            # Build new `DataBuilder`s and `DataManager`s for the split data.
            for split, content in contents.items():
                setattr(self, f"_{split}_content", content)

        # Otherwise, raise an error for an invalid type.
        else:
            raise TypeError(
                "Expected either only ints or only floats when generating "
                f"a data split, got {[type(i) for i in arg_dict.values()]}."
            )

    def _is_split_generated(self):
        """Check if a data split has been generated (not necessarily accessed)"""
        return any(getattr(self, f"_{split}_content") is not None for split in ["train", "val", "test"])

    def save_split(self, name, overwrite=False):
        """Saves the current split of data to an internal location.

        This method can be used to save the current split of data to an
        internal file, such that the same split can be later loaded using
        the `load_split` method (for reproducibility). This method will only
        save the actual split of data, not any of the transforms or other
        parameters which have been applied to the loader.

        Parameters
        ----------
        name: str
            The name of the split to save. This name will be used to identify
            the split when loading it later.
        overwrite: bool
            Whether to overwrite an existing split with the same name.
        """
        # Ensure that there exist data splits (train/val/test data).
        if self._train_content is None and self._val_content is None and self._test_content is None:
            raise NotImplementedError("Cannot save a split of data when no " "split has been generated.")

        # Get each of the individual splits, and for semantic segmentation/image
        # classification, remove the full paths and only save the path relative
        # to the dataset root (so only the file and its directory are saved).
        splits = {}
        if self._info.tasks.ml == "image_classification":
            for split in ["train", "val", "test"]:
                contents = getattr(self, f"_{split}_content")
                if contents is not None:
                    contents = {os.path.relpath(c, self.dataset_root): v for c, v in contents.items()}
                splits[split] = contents
        elif self._info.tasks.ml == "semantic_segmentation":
            for split in ["train", "val", "test"]:
                contents = getattr(self, f"_{split}_content")
                if contents is not None:
                    contents = {
                        os.path.relpath(c, self.dataset_root): os.path.relpath(v, self.dataset_root)
                        for c, v in contents.items()
                    }
                splits[split] = contents
        elif self._info.tasks.ml == "object_detection":
            for split in ["train", "val", "test"]:
                contents = getattr(self, f"_{split}_content")
                splits[split] = contents

        # Save the split to the internal location.
        split_dir = os.path.join(SUPER_BASE_DIR, "splits", self.name)
        os.makedirs(split_dir, exist_ok=True)
        path_split = os.path.join(split_dir, f"{name}.json")
        if os.path.exists(path_split):
            if not overwrite:
                raise FileExistsError(f"A split with the name {name} already exists.")
        with open(path_split, "w") as f:
            json.dump(splits, f)

    def load_split(self, name, **kwargs):
        """Loads a previously saved split of data.

        This method can be used to load a previously saved split of data
        if the split was saved using the `save_split` method. This method
        will only load the actual split of data, not any of the transforms
        or other parameters which have been applied to the loader. You can
        use the traditional split accessors (`train_data`, `val_data`, and
        `test_data`) to access the loaded data.

        You can also load a pre-defined split for the dataset by using its name
        (any potential such splits can be found in the dataset info, and are
        derived from the original dataset).

        Parameters
        ----------
        name: str
            The name of the split to load. This name will be used to identify
            the split to load.
        """
        if kwargs.get("manual_split_set", False):
            splits = kwargs["manual_split_set"]

        else:
            # Ensure that the split exists.
            split_dir = os.path.join(SUPER_BASE_DIR, "splits", self.name)
            if not os.path.exists(os.path.join(split_dir, f"{name}.json")):
                split_dir = os.path.join(self.dataset_root, ".splits")
                if not os.path.exists(os.path.join(split_dir, f"{name}.json")):
                    raise FileNotFoundError(f"Could not find a split with the name {name}.")

            # Load the split from the internal location.
            with open(os.path.join(split_dir, f"{name}.json"), "r") as f:
                splits = json.load(f)

        # Set the split contents.
        for split, content in splits.items():
            # If the data is for image classification or semantic segmentation,
            # then we need to re-construct the full paths to the images.
            if len(content) > 0:
                first_item = list(content.items())[0]
                if not os.path.isabs(first_item[0]):  # backwards compatibility
                    if self._info.tasks.ml == "image_classification":
                        content = {os.path.join(self.dataset_root, c): v for c, v in content.items()}
                    elif self._info.tasks.ml == "semantic_segmentation":
                        content = {
                            os.path.join(self.dataset_root, c): os.path.join(self.dataset_root, v)
                            for c, v in content.items()
                        }
            else:
                content = None

            setattr(self, f"_{split}_content", content)

    def batch(self, batch_size=None):
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
        self._manager.batch_data(batch_size=batch_size)

    def resize_images(self, image_size=None, method="bilinear"):
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
        5. `auto`: Dynamically selects an image size based on a few factors.
           For example, if there are certain outliers in a dataset which are
           of a different size while the majority remain the same, then, the
           behavior of this method is chosen by a majority threshold of the
           sizes of all the images in the dataset. If no shape can be
           inferenced, it returns a default size of (512, 512).

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
        If a transform pipeline is provided, images will be resized
        *before* being passed into the transform pipeline.
        """
        self._manager.assign_resize(image_size=image_size, method=method)

    def transform(
        self,
        transform=NoArgument,
        target_transform=NoArgument,
        dual_transform=NoArgument,
    ):
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
        self._manager.push_transforms(
            transform=transform,
            target_transform=target_transform,
            dual_transform=dual_transform,
        )

    def normalize_images(self, method="scale"):
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
        if method not in ["scale", "imagenet", "standard", None]:
            raise ValueError(f"Received invalid normalization method: '{method}'.")

        if method == "scale":
            normalization_params = "scale"
        elif method == "imagenet":
            normalization_params = "imagenet"
        elif method == "standard":
            normalization_params = self._info.image_stats
        else:
            normalization_params = None

        self.transform(transform=("normalize", normalization_params))

    def labels_to_one_hot(self, add=True):
        """Converts image classification numerical labels to one-hot labels.

        This is a convenience method to apply one-hot vector transformations
        to the output labels for image classification. Essentially, if we have
        a set of labels, [1, 2], it will convert it to [[0, 1, 0], [0, 0, 1]].
        This is a more commonly used format for image classification.

        Parameters
        ----------
        add : bool
            If set to `None` or `False`, this will remove the one-hot
            label transformation from the manager. This is `True` by default,
            which adds the one-hot label transformation.

        """
        if self._info.tasks.ml != "image_classification":
            raise RuntimeError("The `one_hot` label transformation can only " "be used for image classification tasks.")

        self.transform(target_transform=("one_hot", self._info.num_classes, add))

    def mask_to_channel_basis(self, add=True):
        """Converts semantic segmentation masks to channel-wise.

        This is a convenience method to convert integer-labeled semantic
        segmentation masks into channel-by-channel masks, essentially
        one-hot vector transformation but for semantic segmentation. Note
        that if the task is binary segmentation, e.g. there is only one
        class, then this method will do nothing.

        This method should traditionally be called *after* applying general
        transformations to the loader, in order to prevent any issues.

        Parameters
        ----------
        add : bool
            If set to `None` or `False`, this will remove the one-hot
            label transformation from the manager. This is `True` by default,
            which adds the one-hot label transformation.
        """
        if self._info.tasks.ml != "semantic_segmentation":
            raise ValueError(
                "The `mask_to_channel_basis` transformation " "can only be used for semantic segmentation tasks."
            )

        # Warn about binary segmentation tasks.
        if self._info.num_classes == 1:
            log(
                f"No mask-to-channel transformation will be applied for "
                f"a binary segmentation task (dataset {self.name})."
            )
            return

        self.transform(target_transform=("channel_basis", self._info.num_classes, add))

    def generalize_class_detections(self):
        """Generalizes object detection classes to a single class.

        This is a convenience method for object detection tasks, and
        converts all of the individual class labels in the task into
        a single class, essentially allowing the model to purely
        focus on detection of objects and fine-tuning bounding boxes,
        with no focus on differentiating classes of different boxes.

        This method is intended to be used for multi-dataset loaders,
        and will raise an error if using with a single-dataset loader.
        """
        raise ValueError("This method can only be used with multi-dataset loaders.")

    def export_contents(self, export_format=None):
        """Exports the internal contents of the `AgMLDataLoader`.

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
        The raw contents of the dataset.
        """
        return self._builder.export_contents(export_format=export_format)

    def export_tensorflow(self):
        """Exports the contents of the loader in a native TensorFlow dataset.

        This method constructs a `tf.data.Dataset` from the contents of the
        dataset. The dataset maintains the same loading and preprocessing
        pipeline as the actual `DataLoader`, but allows for faster computation
        and integration into a TensorFlow pipeline.

        When constructing the `tf.Data.Dataset`, the `AgMLDataLoader` uses
        the pre-set parameters of the class, including the transforms, image
        resizing, and the training mode. In particular, the tensor conversion
        and automatic batching is done inherently by the `tf.data.Dataset`, but
        transforms can be disabled in the same way they would be in a normal
        `AgMLDataLoader` by running

        > loader.disable_preprocessing()
        > loader.export_tensorflow()

        or potentially

        > loader.eval()
        > loader.export_tensorflow()

        Image resizing is an exception in that if a specific size is not set,
        then it will automatically be set to (512, 512) to prevent errors.

        The same behavior of transforms and image resizing applies also to
        batching. Calling the relevant methods before exporting the dataset
        will result in those methods being applied to the result. The only
        exception is shuffling, since the data is always shuffled upon being
        exported to a `tf.data.Dataset`. This enables better computation.
        Note that if the data is batched, then it is also prefetched.

        Please note, transforms will not be applied if exporting an object
        detection loader. This is due to the impossibility of applying
        transforms to COCO JSON dictionaries and passing them in TensorFlow's
        graph mode. Use `as_keras_sequence` if you want to use transforms.

        Returns
        -------
        A `tf.data.Dataset` enabled to function like the `AgMLDataLoader`, but
        as a native TensorFlow object for TensorFlow pipelines.
        """
        # Update the backend management system.
        from agml.data.exporters.tensorflow import TFExporter

        if get_backend() != "tf":
            if user_changed_backend():
                raise StrictBackendError(change="tf", obj=self.export_tensorflow)
            set_backend("tf")

        # Build the exporter.
        exporter = TFExporter(task=self.info.tasks.ml, builder=self._builder)

        # Update the current state of the loader.
        exporter.assign_state(state=self._manager._train_manager.state)

        # Parse the transforms and resizing for the class.
        transforms = self._manager._transform_manager.get_transform_states()
        resizing = self._manager._resize_manager.size
        exporter.digest_transforms(transforms=transforms, resizing=resizing)

        # Construct and return the loader.
        return exporter.build(batch_size=self._manager._batch_size)

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
        from torch.utils.data import DataLoader

        if get_backend() != "torch":
            if user_changed_backend():
                raise StrictBackendError(change="torch", obj=self.export_torch)
            set_backend("torch")

        # Make a copy of the `AgMLDataLoader` so the following changes
        # don't affect the original loader, just the new one.
        obj = self.copy()

        # Convert to a PyTorch dataset.
        obj.as_torch_dataset()

        # The `DataLoader` automatically batches objects using its
        # own mechanism, so we remove batching from this DataLoader.
        batch_size = loader_kwargs.pop("batch_size", obj._manager._batch_size)
        obj.batch(None)
        shuffle = loader_kwargs.pop("shuffle", obj._manager._shuffle)

        # The `collate_fn` for object detection is different because
        # the COCO JSON dictionaries each have different formats. So,
        # we need to replace it with a custom function.
        collate_fn = loader_kwargs.pop("collate_fn", None)
        if obj.task == "object_detection" and collate_fn is None:
            if any(
                "efficientdet" in i.__class__.__name__.lower()
                for i in self._manager._transform_manager.get_transform_states()["dual_transform"]
            ):
                from agml.backend.tftorch import collate_fn_efficientdet

                collate_fn = collate_fn_efficientdet
            else:
                from agml.backend.tftorch import collate_fn_basic

                collate_fn = collate_fn_basic

        # Return the DataLoader with a copy of this AgMLDataLoader, so
        # that changes to this will not affect the returned loader.
        return DataLoader(
            obj,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

    def export_yolo(self, yolo_path=None):
        """Exports the contents of the loader to the YOLO format, ready for training.

        This method exports the contents of the loader to the YOLO format, and saves
        these contents to a file structure that is ready and compatible for training
        with Ultralytics YOLO models. This method is intended for object detection
        datasets, and raises an error if used with non-object detection datasets.

        Parameters
        ----------
        yolo_path : str
            The path to save the YOLO-formatted dataset to.

        Returns
        -------
        The path to the saved YOLO-formatted dataset.
        """
        if self._info.tasks.ml != "object_detection":
            raise ValueError("The `export_yolo` method can only be used for " "object detection tasks.")
        return export_yolo(self, yolo_path=yolo_path)

    def show_sample(self, image_only=False, no_show=False):
        """Shows a single data sample from the dataset.

        This method generates a data sample from the dataset with an image and
        its corresponding annotation (or, if `image_only` is True, then only the
        image itself). This data sample is then displayed, unless `no_show` is
        True in which case the processed sample will simply be returned.

        Parameters
        ----------
        image_only : optional
            Whether to show only the image or the image and the annotation.
        no_show : optional
            Whether to display the sample or not.

        Returns
        -------
        The data sample with/without annotation.
        """
        # Get the sample (and take only the first one in a batch if batched).
        image, annotations = self[self._manager._get_random_index()]
        if len(image.shape) == 4:
            image = image[0]
            annotations = annotations[0]

        # Show the sample.
        show_sample(self, image_only=image_only, no_show=no_show, sample=(image, annotations))
