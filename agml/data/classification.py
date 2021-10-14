import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from agml.backend.tftorch import set_backend, get_backend
from agml.backend.tftorch import (
    _check_image_classification_transform # noqa
)

from agml.utils.io import get_dir_list, get_file_list
from agml.utils.general import to_camel_case, resolve_list_value

from agml.data.loader import AgMLDataLoader

class AgMLImageClassificationDataLoader(AgMLDataLoader):
    """AgMLDataLoader optimized for image classification datasets.

    Note: This class should never be directly instantiated. Use
    `AgMLDataLoader` and it will auto-dispatch to this class when
    selecting an image classification dataset.
    """
    def __init__(self, dataset, **kwargs):
        # Take care of the `__new__` initialization logic.
        if kwargs.get('skip_init', False):
            return
        super(AgMLImageClassificationDataLoader, self).__init__(dataset, **kwargs)

        # Set up the class data.
        self._load_data_by_directory()

        # Other attributes which may or may not be initialized.
        self._transform_pipeline = None

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
            try:
                item = self._batched_data[indx]
                images, labels = [], []
                for image_path, label in item.items():
                    image = cv2.cvtColor(
                        cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    try:
                        image = self._preprocess_image(image)
                    except TypeError as te:
                        if "PIL" in str(te):
                            raise ValueError(
                                "You need to include torchvision.transforms"
                                ".ToTensor() in a transform pipeline.")
                        else:
                            raise te
                    images.append(image)
                    labels.append(label)
                return images, labels
            except KeyError:
                raise KeyError(
                    f"Index out of range: got {indx}, "
                    f"expected one in range 0 - {len(self)}.")
        else:
            try:
                image_path = self._image_paths[indx]
                label = self._data[image_path]
                image = cv2.cvtColor(
                    cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                try:
                    image = self._preprocess_image(image)
                except TypeError as te:
                    if "PIL" in str(te):
                        raise ValueError(
                            "You need to include torchvision.transforms"
                            ".ToTensor() in a transform pipeline.")
                    else:
                        raise te
                if self._getitem_as_batch:
                    if self._default_image_size != 'default':
                        image = cv2.resize(
                            image, self._default_image_size, cv2.INTER_NEAREST)
                    return np.expand_dims(image, axis = 0), \
                           np.expand_dims(np.array(label), axis = 0)
                return image, label
            except KeyError:
                raise KeyError(
                    f"Index out of range: got {indx}, "
                    f"expected one in range 0 - {len(self)}.")

    def _wrap_reduced_data(self, split):
        """Wraps the reduced class information for `_from_data_subset`."""
        data_meta = getattr(self, f'_{split}_data')
        meta_dict = {
            'name': self.name,
            'data': data_meta,
            'image_paths': list(data_meta.keys()),
            'transform_pipeline': self._transform_pipeline
        }
        return meta_dict

    @classmethod
    def _from_data_subset(cls, meta_dict, meta_kwargs):
        """Initializes the class from a subset of data.

        This method is used internally for the `split` method, and
        generates DataLoaders with a specific subset of the data.
        """
        loader = cls(meta_dict['name'], **meta_kwargs)
        loader._data = meta_dict['data']
        loader._image_paths = meta_dict['image_paths']
        loader._transform_pipeline = meta_dict['transform_pipeline']
        loader._block_split = True
        return loader

    @property
    def labels(self):
        return self._labels

    def _load_data_by_directory(self):
        """Loads image classification data for the `directory_names` format.

        In this format, images are organized by class where the directory
        they are placed in corresponds to their label in the dataset.
        """
        image_label_mapping = {}
        candidate_dirs = get_dir_list(self._dataset_root)
        for dir_ in candidate_dirs:
            if dir_.startswith('.'):
                continue
            dir_path = os.path.join(self._dataset_root, dir_)
            if len(get_file_list(dir_path)) == 0:
                continue
            for file_ in get_file_list(dir_path):
                file_ = os.path.join(dir_path, file_)
                image_label_mapping[file_] = self._info.class_to_num[dir_]
        self._image_paths = list(image_label_mapping.keys())
        self._labels = list(np.unique(list(image_label_mapping.values())))
        self._data = image_label_mapping
        self._reshuffle()

    def _reshuffle(self):
        """Reshuffles the data if allowed to."""
        if not self._shuffle:
            return
        items = list(self._data.items())
        np.random.shuffle(items)
        self._data = dict(items)
        self._image_paths = list(self._data.keys())

    @staticmethod
    def _convert_dict_to_arrays(*dicts):
        """Converts dictionaries mapping images to classes into arrays."""
        arrays = []
        for dict_ in dicts:
            if dict_ is None:
                continue
            fpaths, labels = dict_.keys(), dict_.values()
            arrays.append([np.array(list(fpaths)),
                           np.array(list(labels))])
        return resolve_list_value(tuple(arrays))

    @staticmethod
    def _convert_nested_dict_to_arrays(*dicts):
        """A version of `_convert_dict_to_arrays` optimized for batches."""
        arrays = []
        for dict_ in dicts:
            fpaths, labels = [], []
            for batch in dict_:
                fpaths.append(np.array(list(batch.keys())))
                labels.append(np.array(list(batch.values())))
            fpaths, labels = \
                np.array(fpaths, dtype = object), \
                np.array(labels, dtype = object)
            arrays.append([fpaths, labels])
        return resolve_list_value(tuple(arrays))

    def _preprocess_image(self, image):
        """Preprocesses an image with transformations/other methods."""
        if self._preprocessing_enabled:
            if self._transform_pipeline is not None:
                image = self._transform_pipeline(image)
        return image

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
                        {k: int(v) for k, v in np.array(
                            list(self._data.items()))[split_1]})
                setattr(self, split_names[1],
                        {k: int(v) for k, v in np.array(
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
                            {k: int(v) for k, v in np.array(
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
                        {k: int(v) for k, v in np.array(
                            list(self._data.items()))[split_1]})
                setattr(self, split_names[1],
                        {k: int(v) for k, v in np.array(
                            list(self._data.items()))[split_2]})
            else:
                split_1 = tts[:splits[0]]
                split_2 = tts[splits[0]: splits[0] + splits[1]]
                split_3 = tts[splits[0] + splits[1]:]
                for name, dec_split in zip(split_names, [split_1, split_2, split_3]):
                    setattr(self, name,
                            {k: int(v) for k, v in np.array(
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
            {k: int(v) for k, v in batch} for batch in batches]
        if batch_size != 1:
            self._is_batched = True
        return self._convert_nested_dict_to_arrays(self._batched_data)

    def export_contents(self, as_dict = False):
        """Exports arrays containing the image paths and corresponding labels.

        This method exports the internal contents of the actual dataset: a
        mapping between image file paths and their image classification labels.
        Data can then be processed as desired, and used in other pipelines.

        Parameters
        ---------
        as_dict : bool
            If set to true, then this method will return one dictionary with
            a mapping between image paths and classification labels. Otherwise,
            it returns two arrays with the paths and labels.

        Returns
        -------
        The data contents of the loader.
        """
        if as_dict:
            return self._data
        return self._convert_dict_to_arrays(self._data)

    def transform(self, *, transform = None):
        """Applies vision transformations to the input data.

        This method creates a transformation pipeline which is applied to
        the input image data. It can consist of a number of different items:

        1. A set of `torchvision.transforms`.
        2. A `keras.models.Sequential` model with preprocessing layers.
        3. A method which takes in one input array (the unprocessed
           image) and returns one output array (the processed image).
        4. `None` to remove all transformations.

        Parameters
        ----------
        transform : Any
            Any of the above cases.
        """
        _check_image_classification_transform(transform)
        self._transform_pipeline = transform

    def torch(self, *, image_size = (512, 512), transform = None, **loader_kwargs):
        """Returns a PyTorch DataLoader with this dataset's content.

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
            One of the following:
                1. A set of `torchvision.transforms`.
                2. A method which takes in one input array (the
                   unprocessed image) and returns one output
                   array (the processed image).
                3. `None` to remove all transformations.
            See `AgMLDataLoader.transform()` for more information.
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
        _check_image_classification_transform(transform)
        if get_backend() != 'torch':
            raise ValueError(
                "Using a non-PyTorch transform for `AgMLDataLoader.torch()`.")

        # Create the simplified `torch.utils.data.Dataset` subclass.
        class _DummyDataset(Dataset):
            def __init__(self, image_label_mapping, transform = None): # noqa
                self._mapping = image_label_mapping
                self._transform = transform

            def __len__(self):
                return len(self._mapping)

            def __getitem__(self, indx):
                image, label = list(self._mapping.items())[indx]
                image = cv2.cvtColor(
                    cv2.imread(image), cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size, cv2.INTER_NEAREST)
                if self._transform is not None:
                    image = self._transform(image)
                return image, label

        # Construct the DataLoader.
        _DummyDataset.__name__ = to_camel_case(
            f"{self._info.name}_dataset")
        transform_ = None
        if self._preprocessing_enabled:
            transform_ = transform \
                if transform is not None \
                else (self._transform_pipeline
                      if self._transform_pipeline is not None else None)
        return DataLoader(_DummyDataset(
            self._data, transform = transform_), **loader_kwargs)

    def tensorflow(self, image_size = (512, 512), transform = None):
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
            One of the following:
                1. A `keras.models.Sequential` model with
                   preprocessing layers.
                2. A method which takes in one input array (the
                   unprocessed image) and returns one output
                   array (the processed image).
                3. `None` to remove all transformations.
            See `AgMLDataLoader.transform()` for more information.

        Returns
        -------
        A configured `tf.data.Dataset` with the data.
        """
        import tensorflow as tf
        set_backend('tensorflow')
        _check_image_classification_transform(transform)
        if get_backend() != 'tensorflow':
            raise ValueError(
                "Using a non-TensorFlow transform for `AgMLDataLoader.tensorflow()`.")

        # Create the dataset object with relevant preprocessing.
        @tf.function
        def _image_load_preprocess_fn(path, label):
            image = tf.image.decode_jpeg(tf.io.read_file(path))
            image = tf.cast(tf.image.resize(image, image_size), tf.int32)
            return image, label
        images, labels = self._convert_dict_to_arrays(self._data)
        images, labels = tf.constant(images), tf.constant(labels)
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.shuffle(len(images))
        ds = ds.map(_image_load_preprocess_fn)
        if transform is not None:
            @tf.function
            def _map_preprocessing_fn(image, label):
                return transform(image), label
            ds = ds.map(_map_preprocessing_fn)
        return ds

