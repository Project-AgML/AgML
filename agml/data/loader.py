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

from agml.utils.downloads import download_dataset
from agml.utils.general import resolve_list_value
from agml.backend.config import default_data_save_path
from agml.backend.tftorch import _swap_loader_mro, tf, torch # noqa
from agml.data.metadata import DatasetMetadata

class AgMLDataLoader(object):
    """Loads and processes a dataset from the AgML public repository.

    An `AgMLDataLoader` contains data from one public data source in
    the AgML public repository, and exposes methods allowing easy
    usage both within an AgML pipeline and within an existing pipeline
    in other frameworks like TensorFlow and PyTorch.

    The class should be instantiated with one dataset name, which
    will in turn be downloaded (and simply loaded in subsequent runs).
    Images and annotations can be accessed from relevant properties and
    methods, and transforms from multiple backends can be applied.

    The actual data content within the dataset can be exported in one
    of two ways: using `export_contents()` to get a dictionary with the
    image paths and relevant annotations (too expensive to pre-load in
    all of the images), or using one of the specific methods designed
    for backend integration, namely `torch()` to get a pre-instantiated
    `torch.utils.data.DataLoader` with the dataset pre-loaded in, or
    `tensorflow()` to get a `tf.data.Dataset` with the dataset loaded
    in. See the relevant guides and the method documentations for
    additional info on how to use these methods.

    Notes
    -----
    See the derived classes `AgMLImageClassificationDataLoader`,
    `AgMLObjectDetectionDataLoader`, and `AgMLSemanticSegmentationDataLoader`
    for information on the specific functionalities of the data loader
    for the different types of ML tasks supported in AgML.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load.
    kwargs : Any
        Other initialization parameters. Can include:
        dataset_path : str
            The desired local path to the dataset.
        overwrite : str
            Whether to re-download an existing dataset.
    """

    def __new__(cls, dataset, **kwargs):
        if kwargs.get('skip_init', False):
            return super(AgMLDataLoader, cls).__new__(cls)
        ml_task_type = DatasetMetadata(dataset).tasks.ml
        if ml_task_type == 'image_classification':
            from agml.data.classification \
                import AgMLImageClassificationDataLoader
            return AgMLImageClassificationDataLoader(
                dataset, skip_init = True, **kwargs)
        elif ml_task_type == 'semantic_segmentation':
            from agml.data.segmentation \
                import AgMLSemanticSegmentationDataLoader
            return AgMLSemanticSegmentationDataLoader(
                dataset, skip_init = True, **kwargs)
        elif ml_task_type == 'object_detection':
            from agml.data.detection \
                import AgMLObjectDetectionDataLoader
            return AgMLObjectDetectionDataLoader(
                dataset, skip_init = True, **kwargs)
        raise ValueError(
            f"Unsupported ML task: {ml_task_type}. Please report this error.")

    def __init__(self, dataset, **kwargs):
        # Setup and find the dataset.
        self._info = DatasetMetadata(dataset)
        self._find_dataset(**kwargs)

        # Process internal logic.
        self._shuffle = True
        if not kwargs.get('shuffle', True):
            self._shuffle = False

        # Parameters that may or may not be changed. If not, they are
        # here just for consistency in the class and internal methods.
        self._is_batched = False
        self._batched_data = None

        self._preprocessing_enabled = True
        self._eval_mode = True
        self._preprocessing_store_dict = {}

        self._training_data = None
        self._validation_data = None
        self._test_data = None

        self._getitem_as_batch = False

        self._image_resize = None

        self._tensor_convert_default = lambda *args: resolve_list_value(args)
        self._tensor_convert = self._tensor_convert_default

    @abc.abstractmethod
    def __len__(self):
        """Returns the number of images in the dataset."""
        if self._is_batched:
            return len(self._batched_data)
        return self.num_images

    def __str__(self):
        return repr(self)

    def __repr__(self):
        fmt = (f"<AgMLDataLoader [{self._info.name}] "
               f"[task = {self._info.tasks.ml}]")
        if hasattr(self, '_split_name'):
            fmt += f" [split = {getattr(self, '_split_name')}]"
        fmt += ">"
        return fmt

    def __iter__(self):
        for indx in range(len(self)):
            yield self[indx]

    def __getattribute__(self, item):
        if 'data' in item and any(
                i in item for i in ['training', 'validation', 'test']):
            if getattr(self, '_block_split', False):
                raise AttributeError(
                    "Cannot access split data of already split dataset.")
        return super(AgMLDataLoader, self).__getattribute__(item)

    def _find_dataset(self, **kwargs):
        """Searches for or downloads the dataset in this loader."""
        self._stored_kwargs_for_init = kwargs
        overwrite = kwargs.get('overwrite', False)
        if 'dataset_path' in kwargs:
            kwargs['dataset_path'] = os.path.realpath(
                os.path.expanduser(kwargs['dataset_path']))
        if not overwrite:
            if kwargs.get('dataset_path', False):
                path = kwargs.get('dataset_path')
                if (os.path.basename(path) == self._info.name
                    and os.path.exists(path)) or \
                        os.path.exists(os.path.join(path, self._info.name)):
                    if os.path.basename(path) != self._info.name:
                        path = os.path.join(path, self._info.name)
                    self._dataset_root = path
                    return
            elif os.path.exists(os.path.join(
                    default_data_save_path(), self._info.name)):
                self._dataset_root = os.path.join(
                    default_data_save_path(), self._info.name)
                return
        else:
            if kwargs.get('dataset_path', False):
                path = kwargs.get('dataset_path')
                if (os.path.basename(path) == self._info.name
                    and os.path.exists(path)) or \
                        os.path.exists(os.path.join(path, self._info.name)):
                    if os.path.basename(path) != self._info.name:
                        path = os.path.join(path, self._info.name)
                    print(f'[AgML Download]: Overwriting dataset at '
                          f'{os.path.join(path)}')
                    if os.path.basename(path) != self._info.name:
                        path = os.path.join(path, self._info.name)
                    self._dataset_root = path
            elif os.path.exists(os.path.join(
                    default_data_save_path(), self._info.name)):
                self._dataset_root = os.path.join(
                    default_data_save_path(), self._info.name)
                print(f'[AgML Download]: Overwriting dataset at '
                      f'{os.path.join(self._dataset_root)}')
        if kwargs.get('dataset_path', False):
            download_path = kwargs['dataset_path']
            if os.path.basename(download_path) != self._info.name:
                download_path = os.path.join(download_path, self._info.name)
        else:
            download_path = os.path.join(default_data_save_path(), self._info.name)
        print(f"[AgML Download]: Downloading dataset `{self._info.name}` "
              f"to {download_path}.")
        download_dataset(self._info.name, download_path)
        self._dataset_root = download_path

    @property
    def name(self):
        return self._info.name

    @property
    def num_images(self):
        return self._info.num_images

    @property
    def dataset_root(self):
        # The local path to the dataset.
        return self._dataset_root

    @property
    def info(self):
        """Contains metadata information about the dataset.

        This property returns a `DatasetMetadata` object which
        contains a number of properties with information about
        the dataset. See `DatasetMetadata` for more information.
        """
        return self._info

    def shuffle(self):
        """Shuffles the data in this loader."""
        self._reshuffle()

    def disable_preprocessing(self):
        """Disables internal preprocessing for `__getitem__`.

        By default, images loaded from this dataset when using `__getitem__`
        will apply a transformation or general preprocessing pipeline if
        provided. This method disables these pipelines and returns just the
        original image and its label from the `__getitem__` method.

        To re-enable preprocessing, run `enable_preprocessing()`.
        """
        self._preprocessing_store_dict['_getitem_as_batch'] \
            = self._getitem_as_batch
        self._preprocessing_store_dict['_tensor_convert'] \
            = self._tensor_convert
        self._preprocessing_enabled = False
        self._eval_mode = False
        self._getitem_as_batch = False
        self._tensor_convert = self._tensor_convert_default

    def enable_preprocessing(self):
        """Re-enables internal processing for `__getitem__`.

        If the preprocessing pipeline in the dataset is disabled using
        `disable_preprocessing()`, this method re-enables preprocessing.
        """
        self._preprocessing_enabled = True
        self._eval_mode = False
        self._getitem_as_batch = \
            self._preprocessing_store_dict.get(
                '_getitem_as_batch', False)
        self._tensor_convert = \
            self._preprocessing_store_dict.get(
                '_tensor_convert', self._tensor_convert)

    def resize_images(self, shape = None):
        """Toggles resizing of images when accessing from the loader.

        Passing `shape` in this method will, if preprocessing is enabled,
        automatically resize the images to `shape`. If image resizing
        has been enabled, then passing `None` in place of `shape` will
        disable the resizing and return images in their original shape.

        If shape is set to 'auto', then the loader will attempt to
        automatically inference the shape by checking the shape of some
        random images in the dataset and then setting the shape if they
        all are the same. Note that there may be certain outliers in the
        dataset which are unaccounted for, this does not guarantee to work.
        If the inferencing fails, this falls back to (512, 512).

        Parameters
        ----------
        shape: {list, tuple, str}
            A two-value list or tuple with the new shape, 'auto', or None.
        """
        if shape is not None:
            if shape == 'auto':
                self._auto_inference_shape()
                return
            if not len(shape) == 2:
                msg = f"Expected a two-value tuple with image " + \
                      f"height and width, got {shape}. "
                if len(shape) == 3:
                    msg += f"It appears that you've added a '{shape[-1]}' to " \
                           f"indicate the number of channels. Remove this."
                raise ValueError(msg)
        self._image_resize = shape

    @abc.abstractmethod
    def _auto_inference_shape(self, *args, **kwargs):
        raise NotImplementedError()

    def eval(self):
        """Turns on evaluation mode for the loader.

        Using `eval` is similar to using `disable_preprocessing`, however
        it keeps the resizing of the images (but disables the image batching).
        This is mainly intended to be used when evaluating models, e.g. you
        want to test the model but use traditional (non-augmented) images.

        To re-enable preprocessing, use `enable_preprocessing()`.

        Notes
        -----
        Note that `enable_preprocessing` and `disable_preprocessing` have
        precedence over `eval()`. If you use `enable_preprocessing`, it will
        enable all preprocessing, and if you use `disable_preprocessing`, then
        it will disable all preprocessing. All preprocessing here refers to
        transformation, batching, and image resizing.
        """
        self.disable_preprocessing() # to run/store necessary checks.
        self._eval_mode = True

    @property
    def training_data(self):
        if self._training_data is not None:
            ret_cls = self.__class__._init_from_meta(
                self._wrap_reduced_data('training'))
            ret_cls._split_name = 'train'
            return ret_cls
        raise NotImplementedError(
            "You need to run `split()` with a nonzero 'train' "
            "parameter to use the `training_data` property.")

    @property
    def validation_data(self):
        if self._validation_data is not None:
            ret_cls = self.__class__._init_from_meta(
                self._wrap_reduced_data('validation'))
            ret_cls._split_name = 'validation'
            return ret_cls
        raise NotImplementedError(
            "You need to run `split()` with a nonzero 'val' "
            "parameter to use the `validation_data` property.")

    @property
    def test_data(self):
        if self._test_data is not None:
            ret_cls = self.__class__._init_from_meta(
                self._wrap_reduced_data('test'))
            ret_cls._split_name = 'test'
            return ret_cls
        raise NotImplementedError(
            "You need to run `split()` with a nonzero 'test' "
            "parameter to use the `test_data` property.")

    def as_keras_sequence(self):
        """Makes the `AgMLDataLoader` inherit from `keras.utils.Sequence`.

        This allows the AgMLDataLoader to be directly used in a `model.fit()`
        training pipeline in Keras. This method also enables the dataloader
        to return all items as batches, e.g. if a single image is returned,
        it will still be returned as a batch of 1 for model compatibility.
        """
        _swap_loader_mro(self, 'tf')
        if self._preprocessing_enabled:
            self._getitem_as_batch = True
        self._push_post_getitem('tf')

    def as_torch_dataset(self):
        """Makes the `AgMLDataLoader` inherit from `torch.utils.data.Dataset`.

        This allows the AgMLDataLoader to be directly used in a model
        training pipeline in PyTorch. This method also enables the dataloader
        to return all items as batches, e.g. if a single image is returned,
        it will still be returned as a batch of 1 for model compatibility.
        """
        _swap_loader_mro(self, 'torch')
        if self._preprocessing_enabled:
            self._getitem_as_batch = True
        self._push_post_getitem('torch')

    def on_epoch_end(self):
        # Used for a Keras Sequence
        self._reshuffle()

    @abc.abstractmethod
    def _push_post_getitem(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _wrap_reduced_data(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def _init_from_meta(cls, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _set_state_from_meta(self, *args, **kwargs):
        raise NotImplementedError()

    #### UTILITIES FOR PICKLING ####

    def __getnewargs__(self):
        return self._info.name,

    def __getstate__(self):
        return self._wrap_reduced_data()

    def __setstate__(self, state):
        self.__init__(state['name'], **state['init_kwargs'])
        self._set_state_from_meta(state)

    #### UTILITIES FOR COPYING #####

    def __copy__(self):
        return self.__class__._init_from_meta(
            self._wrap_reduced_data())

    def copy(self):
        import copy as _copy
        return _copy.deepcopy(self)

    #### API METHODS - OVERWRITTEN BY DERIVED CLASSES ####

    @abc.abstractmethod
    def _reshuffle(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def split(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def batch(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def export_contents(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def torch(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def tensorflow(self, *args, **kwargs):
        raise NotImplementedError()





