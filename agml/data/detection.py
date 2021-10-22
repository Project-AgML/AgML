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
import json

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from agml.backend.tftorch import set_backend, get_backend
from agml.backend.tftorch import (
    _check_object_detection_transform, _convert_image_to_torch # noqa
)

from agml.utils.io import get_file_list
from agml.utils.general import to_camel_case

from agml.data.loader import AgMLDataLoader

class AgMLObjectDetectionDataLoader(AgMLDataLoader):
    """AgMLDataLoader optimized for object detection datasets.

    Note: This class should never be directly instantiated. Use
    `AgMLDataLoader` and it will auto-dispatch to this class when
    selecting an object detection dataset.
    """
    def __init__(self, dataset, **kwargs):
        # Take care of the `__new__` initialization logic.
        if kwargs.get('skip_init', False):
            return
        super(AgMLObjectDetectionDataLoader, self).__init__(dataset, **kwargs)

        # Build the data.
        self._find_images_and_annotations()

        # Other attributes which may or may not be initialized.
        self._transform_pipeline = None
        self._dual_transform_pipeline = None

    def __len__(self):
        """Returns the number of images or batches in the dataset."""
        if self._is_batched:
            return len(self._batched_data)
        return len(self._coco_annotation_map)

    def __getitem__(self, indx):
        """Returns one element or one batch of data from the loader."""
        # Different cases for batched vs. non-batched data.
        if isinstance(indx, slice):
            content_length = len(self._coco_annotation_map)
            if self._is_batched:
                content_length = len(self._batched_data)
            contents = range(content_length)[indx]
            return [self[c] for c in contents]
        if self._is_batched:
            item = self._batched_data[indx]
            images, annotations = [], []
            for image_path, annotation in item.items():
                image_path = os.path.join(
                    self._dataset_root, 'images', image_path)
                image = cv2.cvtColor(
                    cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                annotation = self._stack_annotations(annotation)
                image, annotation = self._preprocess_data(image, annotation)
                images.append(image)
                annotations.append(annotation)
            return images, annotations
        else:
            try:
                image_path, annotations = list(
                    self._coco_annotation_map.items())[indx]
                image_path = os.path.join(
                    self._dataset_root, 'images', image_path)
                image = cv2.cvtColor(
                    cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                annotations = self._stack_annotations(annotations)
                image, annotations = self._preprocess_data(image, annotations)
                if self._getitem_as_batch:
                    return np.expand_dims(image, axis = 0), annotations
                return image, annotations
            except KeyError:
                raise KeyError(
                    f"Index out of range: got {indx}, "
                    f"expected one in range 0 - {len(self)}.")

    def _wrap_reduced_data(self, split = None):
        """Wraps the reduced class information for `_init_from_meta`."""
        if split is not None:
            data_meta = getattr(self, f'_{split}_data')
        else:
            data_meta = self._coco_annotation_map
        meta_dict = {
            'name': self.name,
            'coco_annotation_map': data_meta,
            'transform_pipeline': self._transform_pipeline,
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
        loader._coco_annotation_map = meta_dict['coco_annotation_map']
        loader._transform_pipeline = meta_dict['transform_pipeline']
        loader._dual_transform_pipeline = meta_dict['dual_transform_pipeline']
        loader._image_resize = meta_dict['image_resize']
        loader._block_split = True
        return loader

    def _set_state_from_meta(self, meta_dict):
        """Like `_init_from_meta`, but modifies inplace."""
        self._coco_annotation_map = meta_dict['coco_annotation_map']
        self._transform_pipeline = meta_dict['transform_pipeline']
        self._dual_transform_pipeline = meta_dict['dual_transform_pipeline']
        self._image_resize = meta_dict['image_resize']
        if meta_dict['split_name']:
            self._block_split = True
            self._split_name = meta_dict['split_name']

    @property
    def labels(self):
        return self._labels

    def _find_images_and_annotations(self):
        """Finds the image paths and COCO annotation JSON file."""
        self._images = get_file_list(os.path.join(self._dataset_root, 'images'))
        with open(os.path.join(self._dataset_root, 'annotations.json')) as f:
            self._coco_annotations = json.load(f)
        categories, labels = self._coco_annotations['categories'], []
        for category in categories:
            labels.append(category['name'])
        self._labels = labels
        self._map_images_with_annotations()
        self._reshuffle()

    def _map_images_with_annotations(self):
        """Builds a mapping between images and bounding box annotations."""
        image_id_mapping = {}
        for img_meta in self._coco_annotations['images']:
            image_id_mapping[img_meta['id']] = img_meta['file_name']
        coco_map = {fname: [] for fname in image_id_mapping.values()}
        for a_meta in self._coco_annotations['annotations']:
            coco_map[image_id_mapping[a_meta['image_id']]].append(a_meta)
        self._coco_annotation_map = coco_map

    def _reshuffle(self):
        """Reshuffles the data if allowed to."""
        if not self._shuffle:
            return
        items = list(self._coco_annotation_map.items())
        np.random.shuffle(items)
        self._coco_annotation_map = dict(items)

    def _preprocess_data(self, image, annotation):
        """Preprocesses images and annotations with transformations/other methods."""
        if self._preprocessing_enabled:
            if (self._transform_pipeline is None and
                    self._dual_transform_pipeline is None):
                return image, annotation
            if self._transform_pipeline is not None:
                return self._transform_pipeline(image), annotation
            if self._dual_transform_pipeline is not None:
                return self._dual_transform_pipeline(image, annotation)
            if self._image_resize is not None:
                image, annotations = \
                    self._resize_image_and_boxes(image, annotation)
        return image, annotation

    @staticmethod
    def _stack_annotations(annotations):
        """Stacks multiple annotations into one dictionary."""
        annotation = {'bboxes': [], 'labels': [], 'area': [],
                      'image_id': "", 'iscrowd': [], 'segmentation': []}
        for a_set in annotations:
            annotation['bboxes'].append(a_set['bbox'])
            annotation['labels'].append(a_set['category_id'])
            annotation['iscrowd'].append(a_set['iscrowd'])
            annotation['segmentation'].append(a_set['segmentation'])
            annotation['area'].append(a_set['area'])
            annotation['image_id'] = a_set['image_id']
        for key, value in annotation.items():
            # Creating nested sequences from ragged arrays (see numpy).
            if key in ['segmentation']:
                out = np.array(value, dtype = object)
            else:
                out = np.array(value)
            if np.isscalar(out):
                out = out.item()
            annotation[key] = out
        return annotation

    def _resize_image_and_boxes(self, image, annotations):
        y_scale, x_scale = image.shape[0:2]
        bboxes = annotations['bboxes']
        processed_bboxes = []
        for bbox in bboxes:
            x1, y1, width, height = bbox
            processed_bboxes.append([
                x1 / x_scale, y1 / y_scale,
                width / x_scale, height / y_scale])
        image = cv2.resize(
            image, self._image_resize, cv2.INTER_NEAREST)
        y_new, x_new = image.shape[0:2]
        final_processed_bboxes = []
        for bbox in processed_bboxes:
            x1, y1, width, height = bbox
            final_processed_bboxes.append([
                x1 * x_new, y1 * y_new,
                width * x_new, height * y_new])
        final_processed_bboxes = \
            np.array(final_processed_bboxes).astype(np.int32)
        annotations['bboxes'] = final_processed_bboxes
        return image, annotations

    def export_coco(self):
        """Exports the dataset contents in the COCO format.

        This method works like `export_contents()`, but instead of
        exporting the bounding box annotations and labels, it exports
        the entire dataset content in the COCO JSON format.

        See https://cocodataset.org/#format-data for information
        on the COCO JSON annotation format and how to use the output
        contents of this method. If you want just the regular object
        detection outputs, use `export_contents()`.

        Returns
        -------
        A tuple containing the dataset image root and the
        dictionary with the COCO JSON annotations.
        """
        return self._dataset_root, self._coco_annotations

    def export_contents(self, as_dict = False):
        """Exports arrays containing the images and annotations.

        This method exports the internal contents of the actual dataset:
        a mapping between image and the COCO annotations (bounding boxes,
        categories, and other info). Data can then be processed as
        desired, and used in other pipelines.

        Parameters
        ---------
        as_dict : bool
            If set to true, then this method will return one dictionary
            with a mapping between image and annotation paths. Otherwise,
            it returns two arrays with the corresponding paths.

        Returns
        -------
        The data contents of the loader.
        """
        return self._coco_annotation_map

    def export_bboxes_and_labels(self, as_dict = False):
        """Exports only the bounding boxes and labels from the data.

        This method is a content exportation method (see `export_contents`
        and `export_coco`). If you don't want the extra COCO annotation
        info and just want the bounding boxes and their class labels, then
        this method exports arrays with the contents.

        By default, it returns three arrays, where the first is the filenames,
        the second is the bounding boxes, and the third is the category
        labels. If `as_dict` is set to true, then it returns a dictionary
        where each filename is mapped to another dictionary which contains
        two values: 'bboxes' and 'labels' with the respective contents.

        Parameters
        ----------
        as_dict : bool
            Returns content dictionaries as stated above.

        Returns
        -------
        Bounding box and label contents of the dataset.
        """
        base_mapping = self.export_contents()
        bboxes, labels = [], []
        for meta in base_mapping.values():
            one_bbox, one_label = [], []
            for set_ in meta:
                one_bbox.append(set_['bbox'])
                one_label.append(set_['category_id'])
            bboxes.append(np.array(one_bbox))
            labels.append(np.array(one_label))

        if as_dict:
            return {k: {'bboxes': b, 'labels': l}
                    for k, b, l in zip(
                    self._images, bboxes, labels)}
        return self._images, bboxes, labels

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
            tts = np.arange(0, len(self._coco_annotation_map))
            split_names = np.array(arg_names)[np.where(splits != 0)[0]]
            splits = splits[np.where(splits != 0)[0]]
            if len(splits) == 1:
                setattr(self, split_names[0],
                        self._coco_annotation_map.copy())
            elif len(splits) == 2:
                split_1, split_2 = train_test_split(
                    tts, train_size = splits[0],
                    test_size = splits[1], shuffle = shuffle)
                setattr(self, split_names[0],
                        {k: v for k, v in np.array(
                            list(self._coco_annotation_map.items()),
                            dtype = object)[split_1]})
                setattr(self, split_names[1],
                        {k: v for k, v in np.array(
                            list(self._coco_annotation_map.items()),
                            dtype = object)[split_2]})
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
                                list(self._coco_annotation_map.items()),
                                dtype = object)[dec_split]})
        elif any([isinstance(i, int) for i in args]):
            args = [0 if arg is None else arg for arg in args]
            if not all([isinstance(i, int) for i in args]):
                raise TypeError(
                    "Got some split parameters as proportions and "
                    "others as whole numbers. Provide one or the other.")
            if not sum(args) == len(self._coco_annotation_map):
                raise ValueError(
                    f"The sum of the total dataset split should be the "
                    f"length of the dataset "
                    f"({len(self._coco_annotation_map)}): got {sum(args)}.")

            # Get the necessary splits. Rather than having a specific case
            # for each of the cases that have different data splits set,
            # we just generalize it and dynamically set the splits.
            if shuffle:
                self._reshuffle()
            tts = np.arange(0, len(self._coco_annotation_map))
            splits = np.array(args)
            split_names = np.array(arg_names)[np.where(splits != 0)[0]]
            splits = splits[np.where(splits != 0)[0]]
            if len(splits) == 1:
                setattr(self, split_names[0],
                        self._coco_annotation_map.copy())
            elif len(splits) == 2:
                split_1, split_2 = tts[:splits[0]], tts[splits[0]]
                setattr(self, split_names[0],
                        {k: v for k, v in np.array(
                            list(self._coco_annotation_map.items()),
                            dtype = object)[split_1]})
                setattr(self, split_names[1],
                        {k: v for k, v in np.array(
                            list(self._coco_annotation_map.items()),
                            dtype = object)[split_2]})
            else:
                split_1 = tts[:splits[0]]
                split_2 = tts[splits[0]: splits[0] + splits[1]]
                split_3 = tts[splits[0] + splits[1]:]
                for name, dec_split in zip(split_names, [split_1, split_2, split_3]):
                    setattr(self, name,
                            {k: v for k, v in np.array(
                                list(self._coco_annotation_map.items()),
                                dtype = object)[dec_split]})

        # Return the splits.
        return self._training_data, self._validation_data, self._test_data

    def batch(self, batch_size = None):
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
        num_splits = len(self._coco_annotation_map) // batch_size
        data_items = np.array(list(
            self._coco_annotation_map.items()), dtype = object)
        overflow = len(self._coco_annotation_map) - num_splits * batch_size
        extra_items = data_items[-overflow:]
        batches = np.array_split(
            np.array(list(self._coco_annotation_map.items())
                     [:num_splits * batch_size], dtype = object),
            num_splits)
        batches.append(extra_items)

        # Convert the batched data internally and return it.
        self._batched_data = [
            {k: v for k, v in batch} for batch in batches]
        if batch_size != 1:
            self._is_batched = True
        return self._batched_data

    def transform(self, *, transform = None, dual_transform = None):
        """Apples vision transforms to the image and annotation data.

        This method constructs a transformation pipeline which is applied
        to the input image data, as well as the output annotations. In
        particular, the following are ways this method can function:

        1. The `transform` argument can be used to apply transformations
           only to the input image. In particular, this is useful for
           visual augmentation pipelines such as random saturation. This
           argument can consist of one of the following:

            a. A set of `torchvision.transforms`.
            b. A `keras.models.Sequential` model with preprocessing layers.
            c. A method which takes in one input array (the unprocessed
               image) and returns one output array (the processed image).
            d. `None` to remove all transformations.

        2. The `dual_transform` argument can be used to apply a transformation
           both the image and annotation (such as in the case of spatial
           augmentation). This argument should be a method which takes in
           two input parameters (the image array and the COCO JSON annotations
           for the image) and outputs those augmented parameters.

        Parameters
        ----------
        transform : Any
            Any of the above cases.
        dual_transform : Any
            Any of the above cases.
        """
        _check_object_detection_transform(transform, dual_transform)
        if dual_transform is not None:
            self._dual_transform_pipeline = dual_transform
            self._transform_pipeline = None
            return
        self._transform_pipeline = transform
        self._dual_transform_pipeline = None

    def torch(self, *, image_size = (512, 512), transform = None,
              dual_transform = None, **loader_kwargs):
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
            See `AgMLObjectDetectionDataLoader.transform()`
            for information.
        dual_transform : Any
            See `AgMLObjectDetectionDataLoader.transform()`
            for information on this argument.
        loader_kwargs : dict
            Any keyword arguments which will be passed to the
            `torch.utils.data.DataLoader`. See its documentation
            for more information on these keywords.

        Returns
        -------
        A configured `torch.utils.data.DataLoader` with the data.
        """
        import torch
        from torch.utils.data import Dataset, DataLoader
        set_backend('torch')
        _check_object_detection_transform(transform, dual_transform)
        if get_backend() != 'torch':
            raise ValueError(
                "Using a non-PyTorch transform for `AgMLDataLoader.torch()`.")

        class _DummyDataset(Dataset):
            def __init__(self, root, coco_mapping,
                         transform = None, dual_transform = None): # noqa
                self._root = root
                self._coco_mapping = coco_mapping
                self._transform_pipeline = transform
                self._dual_transform_pipeline = dual_transform

            def _preprocess_data(self, image, annotation):
                if (self._transform_pipeline is None and
                        self._dual_transform_pipeline is None):
                    return image, annotation
                if self._transform_pipeline is not None:
                    return self._transform_pipeline(image), annotation
                if self._dual_transform_pipeline is not None:
                    return self._dual_transform_pipeline(image, annotation)
                return image, annotation

            @staticmethod
            def _resize_image_and_box(image, annotations):
                y_scale, x_scale = image.shape[0:2]
                for a_set in annotations:
                    x1, y1, width, height = a_set['bbox']
                    a_set['bbox'] = [
                        x1 / x_scale, y1 / y_scale,
                        width / x_scale, height / y_scale]
                image = cv2.resize(image, image_size, cv2.INTER_NEAREST)
                y_new, x_new = image.shape[0:2]
                for a_set in annotations:
                    x1, y1, width, height = a_set['bbox']
                    a_set['bbox'] = [
                        x1 * x_new, y1 * y_new, width * x_new, height * y_new]
                return _convert_image_to_torch(image), annotations

            def __getitem__(self, indx):
                image, annotations = list(self._coco_mapping.items())[indx]
                image = cv2.cvtColor(
                    cv2.imread(os.path.join(self._root, 'images', image)),
                    cv2.COLOR_BGR2RGB)
                image, annotations = \
                    self._resize_image_and_box(image, annotations)
                annotation = {'bboxes': [], 'labels': [], 'area': [],
                              'image_id': "", 'iscrowd': [], 'segmentation': []}
                for a_set in annotations:
                    annotation['bboxes'].append(a_set['bbox'])
                    annotation['labels'].append(a_set['category_id'])
                    annotation['iscrowd'].append(a_set['iscrowd'])
                    annotation['segmentation'].append(a_set['segmentation'])
                    annotation['area'].append(a_set['area'])
                    annotation['image_id'] = a_set['image_id']
                for key, value in annotation.items():
                    annotation[key] = torch.as_tensor(value)
                image, annotation = self._preprocess_data(image, annotation)
                return image, annotation

            def __len__(self):
                return len(self._coco_mapping)

        # There are an arbitrary number of annotations for each image,
        # so we set `collate_fn` to a direct mapping or else it will throw
        # a RuntimeError because the batches are of different sizes.
        _DummyDataset.__name__ = to_camel_case(
            f"{self._info.name}_dataset")
        return DataLoader(_DummyDataset(
            self._dataset_root, self._coco_annotation_map,
            transform = transform, dual_transform = dual_transform),
            collate_fn = lambda x: tuple(zip(*x)), **loader_kwargs)

    def tensorflow(self, *, image_size = (512, 512), transform = None,
                   dual_transform = None):
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
            See `AgMLObjectDetectionDataLoader.transform()`
            for information.
        dual_transform : Any
            See `AgMLObjectDetectionDataLoader.transform()`
            for information on this argument.

        Returns
        -------
        A configured `tf.data.Dataset` with the data.
        """
        # Note on `tf.config.run_functions_eagerly(True)`. To use this specific mode,
        # that is, `loader.tensorflow()` for object detection, a COCO annotation dictionary
        # is returned and processed in the pipeline, and since TensorFlow in graph mode
        # runs into multiple errors, we have to prevent this by running the dataset
        # in eager mode. To change this behavior, you have to write your own dataset.
        import tensorflow as tf
        tf.config.run_functions_eagerly(True)
        set_backend('tensorflow')
        _check_object_detection_transform(transform, dual_transform)
        if get_backend() != 'tensorflow':
            raise ValueError(
                "Using a non-TensorFlow transform for `AgMLDataLoader.tensorflow()`.")

        # Construct a dataset mapping the image and COCO annotations.
        images, annotations = list(self._coco_annotation_map.keys()), \
                              list(self._coco_annotation_map.values())
        images = tf.constant([
            os.path.join(self._dataset_root, 'images', i) for i in images])
        processed_annotations = []
        for a_set in annotations:
            processed_annotations.append(self._stack_annotations(a_set))
        features = {'bboxes': [], 'labels': [], 'area': [],
                    'image_id': [], 'iscrowd': [], 'segmentation': []}
        for a_set in processed_annotations:
            for feature in features.keys():
                features[feature].append(a_set[feature])
        for feature in features.keys():
            features[feature] = tf.ragged.constant(features[feature])
        feature_ds = tf.data.Dataset.from_tensor_slices(features)
        ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(images), feature_ds))
        ds.shuffle(len(images))

        # Map the relevant preprocessing methods.
        def _resize_image_and_bboxes(image, coco_boxes):
            y_scale, x_scale = image.shape[0:2]
            stack_boxes = tf.stack(
                [coco_boxes[:, 0] / x_scale,
                 coco_boxes[:, 1] / y_scale,
                 coco_boxes[:, 2] / x_scale,
                 coco_boxes[:, 3] / y_scale], axis = -1)
            image = tf.cast(tf.image.resize(
                image, image_size), tf.int32)
            y_new, x_new = image.shape[0:2]
            new_stack = tf.cast(tf.stack(
                [stack_boxes[:, 0] * x_new,
                 stack_boxes[:, 1] * y_new,
                 stack_boxes[:, 2] * x_new,
                 stack_boxes[:, 3] * y_new], axis = -1
            ), tf.int32)
            return image, new_stack

        @tf.function
        def _image_coco_load_preprocess_fn(image, coco): # noqa
            image = tf.image.decode_jpeg(tf.io.read_file(image))
            ret_coco = coco.copy()
            for key in coco.keys():
                try:
                    ret_coco[key] = coco[key].to_tensor()
                except: pass
            image, ret_coco_boxes = tf.py_function(
                _resize_image_and_bboxes,
                [image, ret_coco['bboxes']], [tf.int32, tf.int32])
            ret_coco['bboxes'] = ret_coco_boxes
            return image, ret_coco
        ds = ds.map(_image_coco_load_preprocess_fn)
        if dual_transform is not None:
            @tf.function
            def _map_preprocessing_fn(image, coco):
                return dual_transform(image, coco)
        elif transform is not None:
            @tf.function
            def _map_preprocessing_fn(image, coco):
                return transform(image), coco
        else:
            _map_preprocessing_fn = None
        if _map_preprocessing_fn is not None:
            ds = ds.map(_map_preprocessing_fn)
        return ds



