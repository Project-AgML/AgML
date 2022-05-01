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
from decimal import getcontext, Decimal

import numpy as np

from agml.framework import AgMLSerializable
from agml.data.manager import DataManager
from agml.data.builder import DataBuilder
from agml.data.metadata import DatasetMetadata
from agml.utils.general import NoArgument
from agml.backend.tftorch import (
    get_backend, set_backend,
    user_changed_backend, StrictBackendError,
    _add_dataset_to_mro, # noqa
)


class AgMLDataLoader(AgMLSerializable):
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

    Parameters
    ----------
    dataset : str
        The name of the public dataset you want to load. See the helper
        method `agml.data.public_data_sources()` for a list of datasets.
    kwargs : dict, optional
        dataset_path : str, optional
            A custom path to download and load the dataset from.
        overwrite : bool, optional
            Whether to rewrite and re-install the dataset.

    Notes
    -----
    See the methods for examples on how to use an `AgMLDataLoader` effectively.
    """
    serializable = frozenset((
        'info', 'builder', 'manager', 'train_data', 'val_data', 'test_data'))

    def __init__(self, dataset, **kwargs):
        """Instantiates an `AgMLDataLoader` with the dataset."""
        # Set up the dataset and its associated metadata.
        self._info = DatasetMetadata(dataset)

        # The data for the class is constructed in two stages. First, the
        # internal contents are constructed using a `DataBuilder`, which
        # finds and wraps the local data in a proper format.
        self._builder = DataBuilder(
            info = self._info,
            dataset_path = kwargs.get('dataset_path', None),
            overwrite = kwargs.get('overwrite', False)
        )

        # These contents are then passed to a `DataManager`, which conducts
        # the actual loading and processing of the data when called.
        self._manager = DataManager(
            builder = self._builder,
            task = self._info.tasks.ml,
            name = self._info.name,
            root = self._builder.dataset_root
        )

        # If the dataset is split, then the `AgMLDataLoader`s with the
        # split and reduced data are stored as accessible class properties.
        self._train_data = None
        self._val_data = None
        self._test_data = None
        self._is_split = False

    def __len__(self):
        return self._manager.data_length()

    def __getitem__(self, indexes):
        if isinstance(indexes, slice):
            data = np.arange(self._manager.data_length())
            indexes = data[indexes].tolist()
        return self._manager.get(indexes)

    def __iter__(self):
        for indx in range(len(self)):
            yield self[indx]

    def __repr__(self):
        out = f"<AgMLDataLoader: (dataset={self.name}"
        out += f", task={self.task}"
        out += f") at {hex(id(self))}>"
        return out

    def __str__(self):
        return repr(self)

    def copy(self):
        """Returns a deep copy of the data loader's contents."""
        return self.__copy__()

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
    def image_size(self):
        """Returns the determined image size for the loader.

        This is primarily useful when using auto shape inferencing, to
        access what the final result ends up being. Otherwise, it may
        just return `None` or the shape that the user has set.
        """
        return self._manager._resize_manager.size

    def _generate_split_loader(self, contents, split):
        """Generates a split `AgMLDataLoader`."""
        # Check if the data split exists.
        if contents is None:
            raise ValueError(
                f"Attempted to access '{split}' split when "
                f"the data has not been split for '{split}'.")

        # Load a new `DataManager` and update its internal managers
        # using the state of the existing loader's `DataManager`.
        builder = DataBuilder.from_data(
            contents = contents,
            info = self.info,
            root = self.dataset_root)
        current_manager = copy.deepcopy(self._manager.__getstate__())
        current_manager.pop('builder')
        current_manager['builder'] = builder

        # Build the new accessors and construct the `DataManager`.
        accessors = np.arange(len(builder.get_contents()))
        if self._manager._shuffle:
            np.random.shuffle(accessors)
        current_manager['accessors'] = accessors
        batch_size = current_manager.pop('batch_size')
        current_manager['batch_size'] = None
        new_manager = DataManager.__new__(DataManager)
        new_manager.__setstate__(current_manager)

        # After the builder and accessors have been generated, we need
        # to generate a new list of `DataObject`s.
        new_manager._create_objects(
            new_manager._builder, self.task)

        # Batching data needs to be done independently.
        if batch_size is not None:
            new_manager.batch_data(batch_size = batch_size)

        # Instantiate a new `AgMLDataLoader` from the contents.
        loader_state = self.copy().__getstate__()
        loader_state['builder'] = builder
        loader_state['manager'] = new_manager
        cls = super(AgMLDataLoader, self).__new__(AgMLDataLoader)
        cls.__setstate__(loader_state)
        for attr in ['train', 'val', 'test']:
            setattr(cls, f'_{attr}_data', None)
        cls._is_split = True
        return cls

    @property
    def train_data(self):
        """Stores the `train` split of the data in the loader."""
        if isinstance(self._train_data, AgMLDataLoader):
            return self._train_data
        self._train_data = self._generate_split_loader(
            self._train_data, split = 'train')
        return self._train_data

    @property
    def val_data(self):
        """Stores the `val` split of the data in the loader."""
        if isinstance(self._val_data, AgMLDataLoader):
            return self._val_data
        self._val_data = self._generate_split_loader(
            self._val_data, split = 'val')
        return self._val_data

    @property
    def test_data(self):
        """Stores the `test` split of the data in the loader."""
        if isinstance(self._test_data, AgMLDataLoader):
            return self._test_data
        self._test_data = self._generate_split_loader(
            self._test_data, split = 'test')
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
        self._manager.update_train_state('eval')
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
        self._manager.update_train_state(False)
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

        _add_dataset_to_mro(self, 'tf')
        self._manager.update_train_state('tf')
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
        _add_dataset_to_mro(self, 'torch')
        self._manager.update_train_state('torch')
        return self

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
        self._manager.shuffle(seed = seed)
        return self

    def split(self, train = None, val = None, test = None, shuffle = True):
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
        # Check if the data is already split or batched.
        if self._is_split:
            raise ValueError("Cannot split already split data.")
        elif self._manager._batch_size is not None:
            raise ValueError("Cannot split already batched data. "
                             "Split the data before batching.")

        # If no parameters are passed, then don't do anything.
        arg_dict = {'train': train, 'val': val, 'test': test}
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
            getcontext().prec = 4 # noqa
            valid_args = {k: Decimal(v) / Decimal(1) for k, v in valid_args.items()}
            if not sum(valid_args.values()) == Decimal(1):
                raise ValueError(f"Got floats for input splits and expected a sum "
                                 f"of 1, instead got {sum(valid_args.values())}.")

            # Convert the splits from floats to ints. If the sum of the int
            # splits are greater than the total number of data, then the largest
            # split is decreased in order to keep compatibility in usage.
            num_images = self._info.num_images
            proportions = {k: int(v * Decimal(num_images)) for k, v in valid_args.items()}
            if sum(proportions.values()) != num_images:
                diff = sum(proportions.values()) - num_images
                largest_split = list(proportions.keys())[
                    list(proportions.values()).index(
                        max(proportions.values()))]
                proportions[largest_split] = proportions[largest_split] - diff
            valid_args = proportions.copy()

        # Create the actual data splits.
        if all(isinstance(i, int) for i in valid_args.values()):
            # Ensure that the sum of the splits is the length of the dataset.
            if not sum(valid_args.values()) == self._info.num_images:
                raise ValueError(f"Got ints for input splits and expected a sum "
                                 f"equal to the dataset length, {self._info.num_images},"
                                 f"but instead got {sum(valid_args.values())}.")

            # The splits will be generated as sequences of indices.
            generated_splits = {}
            split = np.arange(0, self._info.num_images)
            names, splits = list(valid_args.keys()), list(valid_args.values())

            # Shuffling of the indexes will occur first, such that there is an even
            # shuffle across the whole sample and not just in the individual splits.
            if shuffle:
                np.random.shuffle(split)

            # Create the list of split indexes.
            if len(valid_args) == 1:
                generated_splits[names[0]] = split
            elif len(valid_args) == 2:
                generated_splits = {k: v for k, v in zip(
                    names, [split[:splits[0]], split[splits[0]:]])}
            else:
                split_1 = split[:splits[0]]
                split_2 = split[splits[0]: splits[0] + splits[1]]
                split_3 = split[splits[0] + splits[1]:]
                generated_splits = {k: v for k, v in zip(
                    names, [split_1, split_2, split_3])}

            # Get the actual split contents from the manager. These contents are
            # not `DataObject`s, rather they are simply the actual mapping of
            # data to be passed to a `DataBuilder` when constructing the splits.
            contents = self._manager.generate_split_contents(generated_splits)

            # Build new `DataBuilder`s and `DataManager`s for the split data.
            for split, content in contents.items():
                setattr(self, f'_{split}_data', content)

        # Otherwise, raise an error for an invalid type.
        else:
            raise TypeError(
                "Expected either only ints or only floats when generating "
                f"a data split, got {[type(i) for i in arg_dict.values()]}.")

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
        self._manager.batch_data(
            batch_size = batch_size
        )

    def resize_images(self, image_size = None):
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

        Notes
        -----
        If a transform pipeline is provided, images will be resized
        *before* being passed into the transform pipeline.
        """
        self._manager.assign_resize(
            image_size = image_size
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
        self._manager.push_transforms(
            transform = transform,
            target_transform = target_transform,
            dual_transform = dual_transform
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

        self.transform(
            transform = ('normalize', normalization_params)
        )

    def labels_to_one_hot(self):
        """Converts image classification numerical labels to one-hot labels.

        This is a convenience method to apply one-hot vector transformations
        to the output labels for image classification. Essentially, if we have
        a set of labels, [1, 2], it will convert it to [[0, 1, 0], [0, 0, 1]].
        This is a more commonly used format for image classification.
        """
        if self._info.tasks.ml != 'image_classification':
            raise ValueError("The `one_hot` label transformation can only "
                             "be used for image classification tasks.")

        self.transform(
            target_transform = ('one_hot', self._info.num_classes)
        )

    def export_contents(self, export_format = None):
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
        return self._builder.export_contents(
            export_format = export_format
        )

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
        if get_backend() != 'tf':
            if user_changed_backend():
                raise StrictBackendError(
                    change = 'tf', obj = self.export_tensorflow)
            set_backend('tf')

        # Build the exporter.
        exporter = TFExporter(
            task = self.info.tasks.ml,
            builder = self._builder
        )

        # Update the current state of the loader.
        exporter.assign_state(state = self._manager._train_manager.state)

        # Parse the transforms and resizing for the class.
        transforms = self._manager._transform_manager.get_transform_states()
        resizing = self._manager._resize_manager.size
        exporter.digest_transforms(
            transforms = transforms,
            resizing = resizing
        )

        # Construct and return the loader.
        return exporter.build(batch_size = self._manager._batch_size)

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

        # Convert to a PyTorch dataset.
        self.as_torch_dataset()

        # The `DataLoader` automatically batches objects using its
        # own mechanism, so we remove batching from this DataLoader.
        batch_size = loader_kwargs.pop(
            'batch_size', self._manager._batch_size)
        self.batch(None)
        shuffle = loader_kwargs.pop(
            'shuffle', self._manager._shuffle)

        # The `collate_fn` for object detection is different because
        # the COCO JSON dictionaries each have different formats. So,
        # we need to replace it with a custom function.
        collate_fn = None
        if self.task == 'object_detection':
            def collate_fn(batch):
                images = torch.stack(
                    [i[0] for i in batch], dim = 0)
                coco = tuple(zip(*[i[1] for i in batch]))
                return images, coco

        # Return the DataLoader with a copy of this AgMLDataLoader, so
        # that changes to this will not affect the returned loader.
        return DataLoader(
            self.copy(), # noqa
            batch_size = batch_size,
            shuffle = shuffle,
            collate_fn = collate_fn,
            **loader_kwargs
        )







