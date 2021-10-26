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
import types

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from agml.backend.tftorch import set_backend, get_backend
from agml.backend.tftorch import (
    _check_semantic_segmentation_transform, # noqa
    _postprocess_torch_annotation, # noqa
    _convert_image_to_torch, _multi_tensor_cat # noqa
)
from agml.backend.learn import set_seed

from agml.utils.io import get_file_list
from agml.utils.general import to_camel_case, resolve_list_value
from agml.utils.logging import log

from agml.data.loader import AgMLDataLoader

class AgMLSemanticSegmentationDataLoader(AgMLDataLoader):
    """AgMLDataLoader optimized for image classification datasets.

    Note: This class should never be directly instantiated. Use
    `AgMLDataLoader` and it will auto-dispatch to this class when
    selecting an image classification dataset.
    """
    def __init__(self, dataset, **kwargs):
        # Take care of the `__new__` initialization logic.
        if kwargs.get('skip_init', False):
            return
        super(AgMLSemanticSegmentationDataLoader, self).__init__(dataset, **kwargs)

        # Load internal data.
        self._load_images_and_annotations()

        # Other attributes which may or may not be initialized.
        self._transform_pipeline = lambda x: x
        self._target_transform_pipeline = lambda x: x
        self._dual_transform_pipeline = None

    def __len__(self):
        """Returns the number of images or batches in the dataset."""
        if self._is_batched:
            return len(self._batched_data)
        return len(self._data)

    def __getitem__(self, indx):
        """Returns one element or one batch of data from the loader."""
        # Different cases for batched vs. non-batched data.
        if isinstance(indx, slice):
            content_length = len(self._data)
            if self._is_batched:
                content_length = len(self._batched_data)
            contents = range(content_length)[indx]
            return [self[c] for c in contents]
        if self._is_batched:
            item = self._batched_data[indx]
            images, annotations = [], []
            for image_path, annotation_path in item.items():
                image = cv2.cvtColor(
                    cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                annotation = cv2.imread(annotation_path)
                image, annotation = self._preprocess_data(image, annotation)
                images.append(image)
                annotations.append(annotation)
            if self._getitem_as_batch:
                return _multi_tensor_cat(images), _multi_tensor_cat(annotations)
            return images, annotations
        else:
            try:
                image_path = self._image_paths[indx]
                annotation_path = self._data[image_path]
                image = cv2.cvtColor(
                    cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                annotation = cv2.imread(annotation_path)
                if annotation.shape[-1] == 3:
                    annotation = annotation[:, :, 0]
                image, annotation = self._preprocess_data(image, annotation)
                if self._getitem_as_batch:
                    return self._tensor_convert(
                        np.expand_dims(image, axis = 0),
                        np.expand_dims(np.array(annotation), axis = 0))
                return image, annotation
            except KeyError:
                raise KeyError(
                    f"Index out of range: got {indx}, "
                    f"expected one in range 0 - {len(self)}.")

    def _wrap_reduced_data(self, split = None):
        """Wraps the reduced class information for `_init_from_meta`."""
        if split is not None:
            data_meta = getattr(self, f'_{split}_data')
        else:
            data_meta = self._data
        meta_dict = {
            'name': self.name,
            'data': data_meta,
            'image_paths': list(data_meta.keys()),
            'transform_pipeline': self._transform_pipeline,
            'target_transform_pipeline': self._target_transform_pipeline,
            'dual_transform_pipeline': self._dual_transform_pipeline,
            'image_resize': self._image_resize,
            'init_kwargs': self._stored_kwargs_for_init,
            'split_name': getattr(self, '_split_name', False)
        }
        return meta_dict

    @classmethod
    def _init_from_meta(cls, meta_dict):
        """Initializes the class from a set of metadata."""
        loader = cls(meta_dict['name'], **meta_dict['init_kwargs'])
        loader._data = meta_dict['data']
        loader._image_paths = meta_dict['image_paths']
        loader._transform_pipeline = meta_dict['transform_pipeline']
        loader._target_transform_pipeline = meta_dict['target_transform_pipeline']
        loader._dual_transform_pipeline = meta_dict['dual_transform_pipeline']
        loader._image_resize = meta_dict['image_resize']
        loader._block_split = True
        return loader

    def _set_state_from_meta(self, meta_dict):
        """Like `_init_from_meta`, but modifies inplace."""
        self._data = meta_dict['data']
        self._image_paths = meta_dict['image_paths']
        self._transform_pipeline = meta_dict['transform_pipeline']
        self._target_transform_pipeline = meta_dict['target_transform_pipeline']
        self._dual_transform_pipeline = meta_dict['dual_transform_pipeline']
        self._image_resize = meta_dict['image_resize']
        if meta_dict['split_name']:
            self._block_split = True
            self._split_name = meta_dict['split_name']

    def _auto_inference_shape(self):
        """Attempts to automatically inference the dataset shape."""
        imgs = list(self._image_paths)
        imgs = np.random.choice(imgs, 20)
        shapes = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in imgs]
        if not np.all(shapes == shapes[0]):
            log("Could not inference a constant shape for all "
                "dataset elements. Defaulting to (512, 512).")
            self._image_resize = (512, 512)
        self._image_resize = shapes[0][:2]

    def _load_images_and_annotations(self):
        """Loads semantic segmentation data for the loader.

        Image data is loaded from an `images` directory, and pixel-wise
        annotated images are loaded from an `annotations` directory.
        """
        image_dir = os.path.join(self._dataset_root, 'images')
        annotation_dir = os.path.join(self._dataset_root, 'annotations')
        images, annotations = sorted(get_file_list(image_dir)), \
                              sorted(get_file_list(annotation_dir))
        image_annotation_map = {}
        for image_path, annotation_path in zip(images, annotations):
            image_annotation_map[os.path.join(image_dir, image_path)] \
                = os.path.join(annotation_dir, annotation_path)
        self._image_paths = list(image_annotation_map.keys())
        self._data = image_annotation_map
        self._reshuffle()

    def _reshuffle(self):
        """Reshuffles the data if allowed to."""
        if not self._shuffle:
            return
        items = list(self._data.items())
        np.random.shuffle(items)
        self._data = dict(items) # noqa
        self._image_paths = list(self._data.keys())

    @staticmethod
    def _convert_dict_to_arrays(*dicts):
        """Converts dictionaries mapping images to classes into arrays."""
        arrays = []
        for dict_ in dicts:
            if dict_ is None:
                continue
            img_fpaths, ann_fpaths = dict_.keys(), dict_.values()
            arrays.append([np.array(list(img_fpaths)),
                           np.array(list(ann_fpaths))])
        return resolve_list_value(tuple(arrays))

    @staticmethod
    def _convert_nested_dict_to_arrays(*dicts):
        """A version of `_convert_dict_to_arrays` optimized for batches."""
        arrays = []
        for dict_ in dicts:
            img_fpaths, ann_fpaths = [], []
            for batch in dict_:
                img_fpaths.append(np.array(list(batch.keys())))
                ann_fpaths.append(np.array(list(batch.values())))
            img_fpaths, ann_fpaths = \
                np.array(img_fpaths, dtype = object), \
                np.array(ann_fpaths, dtype = object)
            arrays.append([img_fpaths, ann_fpaths])
        return resolve_list_value(tuple(arrays))

    def _preprocess_data(self, image, annotation):
        """Preprocesses images and annotations with transformations/other methods."""
        if self._preprocessing_enabled:
            if self._image_resize is not None:
                image = cv2.resize(
                    image, self._image_resize, cv2.INTER_NEAREST)
                annotation = cv2.resize(
                    annotation, self._image_resize, cv2.INTER_NEAREST)
            try:
                annotation = np.expand_dims(annotation, axis = -1)
                try:
                    image = self._transform_pipeline(image)
                    annotation = self._target_transform_pipeline(annotation)
                except Exception:
                    raise ValueError(
                        "Encountered an error when using both `transform` and "
                        "`target_transform`. Please check the documentation for "
                        "`AgMLSemanticSegmentationDataLoader.transform()` to "
                        "ensure that you are passing valid transformations.")
                if self._dual_transform_pipeline is not None:
                    if isinstance(self._dual_transform_pipeline, types.FunctionType): # noqa
                        image, annotation = \
                            self._dual_transform_pipeline(image, annotation)
                    else:
                        seed = np.random.randint(2147483647)
                        set_seed(seed)
                        image = self._dual_transform_pipeline(image)
                        set_seed(seed)
                        annotation = self._dual_transform_pipeline(annotation)
                        if get_backend() == 'torch':
                            annotation = _postprocess_torch_annotation(annotation)
                annotation = np.squeeze(annotation)
            except TypeError as te:
                if "PIL" in str(te):
                    raise ValueError(
                        "You need to include torchvision.transforms"
                        ".ToTensor() in a transform pipeline.")
                else:
                    raise te
        elif self._eval_mode:
            if self._image_resize is not None:
                image = cv2.resize(
                    image, self._image_resize, cv2.INTER_NEAREST)
                annotation = cv2.resize(
                    annotation, self._image_resize, cv2.INTER_NEAREST)
        return self._tensor_convert(image, annotation)

    def _push_post_getitem(self, backend):
        if backend == 'tf':
            from agml.backend.tftorch import tf
            self._tensor_convert = lambda image, annotation: \
                (tf.constant(image), tf.constant(annotation))
        elif backend == 'torch':
            self._tensor_convert = lambda image, annotation: \
                (_convert_image_to_torch(image),
                 _convert_image_to_torch(annotation))

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
        shuffle : {bool}
            Whether to shuffle the split data.

        Returns
        -------
        A tuple with the three data splits.
        """
        if self._is_batched:
            raise NotImplementedError(
                "Cannot split data after it has been batched. "
                "Run split before batching, and use the relevant"
                "properties to access the split data.")
        if hasattr(self, '_block_split'):
            raise NotImplementedError(
                "Cannot further split already split data.")

        args = [train, val, test]
        if all([i is None for i in args]):
            self._training_data = None
            self._validation_data = None
            self._test_data = None
            return None, None, None
        arg_names = ['_training_data', '_validation_data', '_test_data']

        # Process splits for proportions or whole numbers.
        if any([isinstance(i, float) for i in args]):
            args = [0.0 if arg is None else arg for arg in args]
            if not all([isinstance(i, float) for i in args]):
                raise TypeError(
                    "Got some split parameters as proportions and "
                    "others as whole numbers. Provide one or the other.")
            if not round(sum(args), 3) == 1.0:
                raise ValueError(
                    f"The sum of the split proportions "
                    f"should be 100% (1.0): got {sum(args)}.")

            # Get the necessary splits. Rather than having a specific case
            # for each of the cases that have different data splits set,
            # we just generalize it and dynamically set the splits.
            if shuffle:
                self._reshuffle()
            splits = np.array(args)
            tts = np.arange(0, len(self._data))
            split_names = np.array(arg_names)[np.where(splits != 0)[0]]
            splits = splits[np.where(splits != 0)[0]]
            if len(splits) == 1:
                setattr(self, split_names[0], self._data.copy())
            elif len(splits) == 2:
                split_1, split_2 = train_test_split(
                    tts, train_size = splits[0],
                    test_size = splits[1], shuffle = shuffle)
                setattr(self, split_names[0],
                        {k: v for k, v in np.array(
                            list(self._data.items()))[split_1]})
                setattr(self, split_names[1],
                        {k: v for k, v in np.array(
                            list(self._data.items()))[split_2]})
            else:
                split_1, split_overflow = train_test_split(
                    tts, train_size = splits[0],
                    test_size = round(splits[1] + splits[2], 2), shuffle = shuffle)
                split_2, split_3 = train_test_split(
                    split_overflow,
                    train_size = splits[1] / (splits[1] + splits[2]),
                    test_size = splits[2] / (splits[1] + splits[2]), shuffle = shuffle)
                for name, dec_split in zip(split_names, [split_1, split_2, split_3]):
                    setattr(self, name,
                            {k: v for k, v in np.array(
                                list(self._data.items()))[dec_split]})
        elif any([isinstance(i, int) for i in args]):
            args = [0 if arg is None else arg for arg in args]
            if not all([isinstance(i, int) for i in args]):
                raise TypeError(
                    "Got some split parameters as proportions and "
                    "others as whole numbers. Provide one or the other.")
            if not sum(args) == len(self._data):
                raise ValueError(
                    f"The sum of the total dataset split should be the "
                    f"length of the dataset ({len(self._data)}): got {sum(args)}.")

            # Get the necessary splits. Rather than having a specific case
            # for each of the cases that have different data splits set,
            # we just generalize it and dynamically set the splits.
            if shuffle:
                self._reshuffle()
            tts = np.arange(0, len(self._data))
            splits = np.array(args)
            split_names = np.array(arg_names)[np.where(splits != 0)[0]]
            splits = splits[np.where(splits != 0)[0]]
            if len(splits) == 1:
                setattr(self, split_names[0], self._data.copy())
            elif len(splits) == 2:
                split_1, split_2 = tts[:splits[0]], tts[splits[0]]
                setattr(self, split_names[0],
                        {k: v for k, v in np.array(
                            list(self._data.items()))[split_1]})
                setattr(self, split_names[1],
                        {k: v for k, v in np.array(
                            list(self._data.items()))[split_2]})
            else:
                split_1 = tts[:splits[0]]
                split_2 = tts[splits[0]: splits[0] + splits[1]]
                split_3 = tts[splits[0] + splits[1]:]
                for name, dec_split in zip(split_names, [split_1, split_2, split_3]):
                    setattr(self, name,
                            {k: v for k, v in np.array(
                                list(self._data.items()))[dec_split]})

        # Return the splits.
        return self._convert_dict_to_arrays(
            self._training_data, self._validation_data, self._test_data)

    def batch(self, batch_size = 8):
        """Combines elements of the data in the loader into batches of data.

        Parameters
        ----------
        batch_size : int
            Represents the size of an individual batch of data. The final
            batch of data will not necessarily of length <= `batch_size`.
            If `batch_size` is set to `None`, then this method will undo
            any batching on data and reset the original data state.

        Returns
        -------
        The batched image paths and corresponding labels.
        """
        # Check for un-batching.
        if batch_size is None:
            self._is_batched = False
            self._batched_data.clear()
            return

        # Get a list of all of the batched data.
        num_splits = len(self._data) // batch_size
        data_items = np.array(list(self._data.items()))
        overflow = len(self._data) - num_splits * batch_size
        extra_items = data_items[-overflow:]
        batches = np.array_split(
            np.array(list(self._data.items())
                     [:num_splits * batch_size]), num_splits)
        batches.append(extra_items)

        # Convert the batched data internally and return it.
        self._batched_data = [
            {k: v for k, v in batch} for batch in batches]
        if batch_size != 1:
            self._is_batched = True
        return self._convert_nested_dict_to_arrays(self._batched_data)

    def transform(self, *, transform = None, target_transform = None,
                  dual_transform = None):
        """Apples vision transforms to the image and annotation data.

        This method constructs a transformation pipeline which is applied
        to the input image data, as well as the output annotations. In
        particular, the following are ways this method can function:

        1. The `transform` and `target_transform` arguments can be used
           to apply transformations to the image and target annotation,
           respectively. These transforms are applied independently, and
           each can consist of any of the following:

            a. A set of `torchvision.transforms`.
            b. A `keras.models.Sequential` model with preprocessing layers.
            c. A method which takes in one input array (the unprocessed
               image) and returns one output array (the processed image).
            d. `None` to remove all transformations.

        2. To apply a single transform pipeline to both the image and
           target annotation, use `dual_transform`. This argument can
           consist of anything from the above list, except for (c), where
           the input and output should be two arrays, and (b), since TensorFlow
           uses different graph-level and operation-level seeds. To use Keras
           preprocessing layers for dual transform, you can use the helper
           method `agml.data.experimental.generate_keras_dual_transform`, see
           that method for information about how to manually write it. Note
           that  if (a), is passed then the pipeline will be applied to both
           the image and the annotation by using the same random seed.

        Parameters
        ----------
        transform : Any
            Any of the above cases.
        target_transform : Any
            Any of the above cases.
        dual_transform : Any
            Any of the above cases.
        """
        _check_semantic_segmentation_transform(
            transform, target_transform, dual_transform)
        self._dual_transform_pipeline = dual_transform
        self._transform_pipeline = transform
        if self._transform_pipeline is None:
            self._transform_pipeline = lambda x: x
        if self._target_transform_pipeline is None:
            self._target_transform_pipeline = lambda x: x

    def export_contents(self, as_dict = False):
        """Exports arrays containing the image and annotation paths.

        This method exports the internal contents of the actual dataset: a
        mapping between image and annotation file paths. Data can then be
        processed as desired, and used in other pipelines.

        Parameters
        ---------
        as_dict : bool
            If set to true, then this method will return one dictionary with
            a mapping between image and annotation paths. Otherwise, it returns
            two arrays with the corresponding paths.

        Returns
        -------
        The data contents of the loader.
        """
        if as_dict:
            return self._data
        return self._convert_dict_to_arrays(self._data)

    def torch(self, image_size = (512, 512), transform = None,
              target_transform = None, dual_transform = None, **loader_kwargs):
        """Returns a PyTorch DataLoader with the dataset's content.

        This method allows the exportation of the data inside this
        loader to a `torch.utils.data.DataLoader`, for direct usage
        inside of a PyTorch pipeline. It first constructs a
        `torch.utils.data.Dataset` with the provided preprocessing
        and then wraps it into a `torch.utils.data.DataLoader`.

        If transforms are provided in `AgMLDataLoader.transform()`,
        those will be used by default (unless preprocessing is
        disabled), unless these are overwritten by a different
        preprocessing pipeline as given in the `preprocessing` argument.

        For greater customization of the dataset workings, use the
        `export_contents()` method to get the actual data itself and
        then construct your own pipeline if necessary.

        Parameters
        ----------
        image_size : tuple
            A tuple of two values containing the output
            image size. This defaults to `(512, 512)`.
        transform : Any
            See `AgMLSemanticSegmentationDataLoader.transform()`
            for information (this works alongside `target_transform`).
        target_transform : Any
            See `AgMLSemanticSegmentationDataLoader.transform()`
            for information (this works alongside `transform`).
        dual_transform : Any
            See `AgMLSemanticSegmentationLoader.transform()`
            for information on this argument.
        loader_kwargs : dict
            Any keyword arguments which will be passed to the
            `torch.utils.data.DataLoader`. See its documentation
            for more information on these keywords.

        Returns
        -------
        A configured `torch.utils.data.DataLoader` with the data.
        """
        from torch.utils.data import Dataset, DataLoader
        set_backend('torch')
        _check_semantic_segmentation_transform(
            transform, target_transform, dual_transform)
        if get_backend() != 'torch':
            raise ValueError(
                "Using a non-PyTorch transform for `AgMLDataLoader.torch()`.")

        # Create the simplified `torch.utils.data.Dataset` subclass.
        class _DummyDataset(Dataset):
            def __init__(self, image_label_mapping,
                         transform = None, target_transform = None, # noqa
                         dual_transform = None): # noqa
                self._mapping = image_label_mapping
                if dual_transform is not None:
                    self._dual_transform_pipeline = dual_transform
                    self._transform_pipeline = transform
                    self._target_transform_pipeline = target_transform
                else:
                    self._transform_pipeline = transform
                    self._target_transform_pipeline = target_transform
                    self._dual_transform_pipeline = None

            def __len__(self):
                return len(self._mapping)

            def _apply_transform(self, image, annotation):
                if self._dual_transform_pipeline is not None:
                    if isinstance(self._dual_transform_pipeline, types.FunctionType):  # noqa
                        image, annotation = \
                            self._dual_transform_pipeline(image, annotation)
                    else:
                        seed = np.random.randint(2147483647)
                        set_seed(seed)
                        image = self._dual_transform_pipeline(image)
                        set_seed(seed)
                        annotation = self._dual_transform_pipeline(annotation)
                        annotation = _postprocess_torch_annotation(annotation)
                if self._transform_pipeline is None \
                        and self._target_transform_pipeline is None:
                    return image, annotation
                elif self._transform_pipeline is not None \
                        and self._target_transform_pipeline is not None:
                    try:
                        image = self._transform_pipeline(image)
                        annotation = self._target_transform_pipeline(annotation)
                    except Exception:
                        raise ValueError(
                            "Encountered an error when using both `transform` and "
                            "`target_transform`. Please check the documentation for "
                            "`AgMLSemanticSegmentationDataLoader.transform()` to "
                            "ensure that you are passing valid transformations.")
                elif self._transform_pipeline is not None \
                        and self._target_transform_pipeline is None:
                    image = self._transform_pipeline(image)
                elif self._transform_pipeline is None \
                        and self._target_transform_pipeline is not None:
                    annotation = self._target_transform_pipeline(annotation)
                return image, annotation

            def __getitem__(self, indx):
                image, annotation = list(self._mapping.items())[indx]
                image = cv2.cvtColor(
                    cv2.imread(image), cv2.COLOR_BGR2RGB)
                annotation = cv2.imread(annotation)
                image = cv2.resize(
                    image, image_size, cv2.INTER_NEAREST)
                annotation = cv2.resize(
                    annotation, image_size, cv2.INTER_NEAREST)
                image, annotation = \
                    _convert_image_to_torch(image), \
                    _convert_image_to_torch(annotation)
                image, annotation = self._apply_transform(image, annotation)
                return image, annotation

        # Construct the DataLoader.
        _DummyDataset.__name__ = to_camel_case(
            f"{self._info.name}_dataset")
        return DataLoader(_DummyDataset(
            self._data, transform = transform,
            target_transform = target_transform,
            dual_transform = dual_transform), **loader_kwargs)

    def tensorflow(self, image_size = (512, 512), transform = None,
                   target_transform = None, dual_transform = None):
        """Returns a `tf.data.Dataset` with this dataset's contents.

        This method allows the exportation of the data inside this
        loader to a `tf.data.Dataset`, for direct and efficient usage
        inside of a TensorFlow pipeline. The internal data representation
        is wrapped into a `tf.data.Dataset`, and its ease of use allows
        data batching, shuffling, prefetching, and even mapping other
        preprocessing functions outside of the provided transformations.

        Unlike the PyTorch equivalent of this method, there are no
        keyword arguments which control dataset batching, shuffling, etc.,
        since TensorFlow's API makes this much simpler and since depending
        on which methods you want to use, the most optimized order will
        change. Trying to account for this logic is inefficient. Note,
        however, that the dataset is automatically shuffled upon creation.

        For greater customization of the dataset workings, use the
        `export_contents()` method to get the actual data itself and
        then construct your own pipeline if necessary.

        Parameters
        ----------
        image_size : tuple
            A tuple of two values containing the output
            image size. This defaults to `(512, 512)`.
        transform : Any
            See `AgMLSemanticSegmentationDataLoader.transform()`
            for information (this works alongside `target_transform`).
        target_transform : Any
            See `AgMLSemanticSegmentationDataLoader.transform()`
            for information (this works alongside `transform`).
        dual_transform : Any
            See `AgMLSemanticSegmentationLoader.transform()`
            for information on this argument.

        Returns
        -------
        A configured `tf.data.Dataset` with the data.
        """
        import tensorflow as tf
        set_backend('tensorflow')
        _check_semantic_segmentation_transform(
            transform, target_transform, dual_transform)
        if get_backend() != 'tensorflow':
            raise ValueError(
                "Using a non-TensorFlow transform for `AgMLDataLoader.tensorflow()`.")

        # Create the dataset object with relevant preprocessing.
        @tf.function
        def _image_load_preprocess_fn(image, annotation):
            image = tf.image.decode_jpeg(tf.io.read_file(image))
            image = tf.cast(tf.image.resize(image, image_size), tf.int32)
            annotation = tf.image.decode_jpeg(tf.io.read_file(annotation))
            annotation = tf.cast(tf.image.resize(annotation, image_size), tf.int32)
            return image, annotation
        images, labels = self._convert_dict_to_arrays(self._data)
        images, labels = tf.constant(images), tf.constant(labels)
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.shuffle(len(images))
        ds = ds.map(_image_load_preprocess_fn)

        # Apply the transforms.
        if all(i is None for i in
               [transform, target_transform, dual_transform]):
            return ds
        if transform is None:
            transform = tf.identity
        if target_transform is None:
            target_transform = tf.identity
        if dual_transform is not None:
            if isinstance(dual_transform, types.FunctionType):  # noqa
                @tf.function
                def _map_preprocess_fn(image_, annotation_):
                    image_ = transform(image_)
                    annotation_ = target_transform(annotation_)
                    return dual_transform(image_, annotation_)
            else:
                @tf.function
                def _map_preprocess_fn(image_, annotation_):
                    image_ = transform(image_)
                    annotation_ = target_transform(annotation_)
                    seed = np.random.randint(2147483647)
                    set_seed(seed); tf.random.set_seed(seed) # noqa
                    image_ = dual_transform(image_)
                    set_seed(seed); tf.random.set_seed(seed) # noqa
                    annotation_ = dual_transform(annotation_)
                    return image_, annotation_
        else:
            @tf.function
            def _map_preprocess_fn(image_, annotation_):
                image_ = transform(image_)
                annotation_ = target_transform(annotation_)
                return image_, annotation_

        ds = ds.map(_map_preprocess_fn)
        return ds



