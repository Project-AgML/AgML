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
import re
import sys
import json

from agml.framework import AgMLSerializable
from agml.backend.config import data_save_path
from agml.utils.downloads import download_dataset
from agml.utils.io import get_file_list, get_dir_list, is_image_file


class DataBuilder(AgMLSerializable):
    """Builds an internal representation format of AgML data.

    This class doesn't affect the state of the `AgMLDataLoader`, but
    instead generates an internal representation of the data content
    of a dataset. This allows for all data to be loaded in a standard
    format and streamlined into the `DataManager`.
    
    Primarily, this class attempts to locate the dataset and if unable
    to do so, it downloads the dataset from the public bucket. Then,
    it creates a mapping between the images and annotations which is
    used by the `DataManager` inside the `AgMLDataLoader`.
    """
    serializable = frozenset(
        ('name', 'task', 'info_map',
         'dataset_root', 'data', 'external_image_sources'))

    def __init__(self, info, dataset_path, overwrite):
        # Attempt to locate or download the dataset.
        self._name = info.name
        self._task = info.tasks.ml
        self._info_map = info.class_to_num
        self._external_image_sources = info.external_image_sources
        self._configure_dataset(
            dataset_path = dataset_path, overwrite = overwrite)
        self._data = None

    @classmethod
    def from_data(cls, contents, info, root):
        """Initializes the `DataBuilder` from a pre-built set of data.

        This is mainly used when running `AgMLDataLoader.split`, to create
        a `DataManager` with a split of data. Functionally, it disables
        all of the actual generation protocols of the `DataBuilder` and
        just pre-assigns all of the values to it.
        """
        obj = super(DataBuilder, cls).__new__(cls)
        obj._name = info.name
        obj._task = info.tasks.ml
        obj._info_map = info.class_to_num
        obj._dataset_root = root
        obj._data = contents
        obj._external_image_sources = info.external_image_sources
        return obj

    @property
    def dataset_root(self):
        return self._dataset_root
        
    def _configure_dataset(self, **kwargs):
        """Finds and configures the dataset into the loader."""
        # Check if the user wants to overwrite the existing dataset,
        # and resolve the potentially provided custom dataset path.
        overwrite = kwargs.get('overwrite', False)
        if kwargs.get('dataset_path', None):
            kwargs['dataset_path'] = os.path.realpath(
                os.path.expanduser(kwargs['dataset_path']))

        # If the user doesn't want to overwrite the existing contents,
        # first check whether the dataset already exists. If so, then set
        # the dataset root and return without doing any downloading.
        #
        # Note that the `dataset_path` is resolved as follows: If the path
        # ends with the name of the dataset, e.g., `/root/datasets/<name>`,
        # then we pop '<name>' from the end of the path in order to prevent
        # the dataset from being downloaded to `/root/datasets/<name>/<name>`.
        if not overwrite:
            if kwargs.get('dataset_path', False):
                path = kwargs.get('dataset_path')
                if (os.path.basename(path) == self._name
                    and os.path.exists(path)) or \
                        os.path.exists(os.path.join(path, self._name)):
                    if os.path.basename(path) != self._name:
                        path = os.path.join(path, self._name)
                    self._dataset_root = path
                    return

            elif os.path.exists(os.path.join(
                    data_save_path(), self._name)):
                self._dataset_root = os.path.join(
                    data_save_path(), self._name)
                return

        # If the user wants to overwrite, or the dataset doesn't exist
        # at the path, then we create the root in the same way, except
        # we just also download the dataset to the path.
        else:
            if kwargs.get('dataset_path', False):
                path = kwargs.get('dataset_path')
                if (os.path.basename(path) == self._name
                    and os.path.exists(path)) or \
                        os.path.exists(os.path.join(path, self._name)):
                    if os.path.basename(path) != self._name:
                        path = os.path.join(path, self._name)
                    print(f'[AgML Download]: Overwriting dataset at '
                          f'{os.path.join(path)}')
                    if os.path.basename(path) != self._name:
                        path = os.path.join(path, self._name)
                    self._dataset_root = path

            elif os.path.exists(os.path.join(
                    data_save_path(), self._name)):
                self._dataset_root = os.path.join(
                    data_save_path(), self._name)
                sys.stderr.write(
                    f'[AgML Download]: Overwriting dataset at '
                    f'{os.path.join(self._dataset_root)}')

        # Performs the actual downloading of the dataset.
        if kwargs.get('dataset_path', False):
            download_path = kwargs['dataset_path']
            if os.path.basename(download_path) != self._name:
                download_path = os.path.join(download_path, self._name)
        else:
            download_path = os.path.join(data_save_path(), self._name)
        sys.stderr.write(
            f"[AgML Download]: Downloading dataset "
            f"`{self._name}` to {download_path}.")
        download_dataset(self._name, download_path)
        self._dataset_root = download_path

    def _generate_contents(self, task):
        """Dispatches to a content generation method for the provided task."""
        if self._data is not None:
            return
        if task == 'image_classification':
            self._generate_image_classification_data()
        elif task == 'image_regression':
            self._generate_image_regression_data()
        elif task == 'object_detection':
            self._generate_object_detection_data()
        else:
            self._generate_semantic_segmentation_data()

    def get_contents(self):
        """Extracts the internal representation of the data content."""
        # Create the internal content representation of the dataset.
        self._generate_contents(self._task)
        return self._data

    def export_contents(self, export_format):
        """Returns the raw contents of the loader."""
        # We start by constructing a default mapping.
        contents = self.get_contents()

        # For a COCO JSON dictionary, we have to make the full paths.
        if self._task == 'object_detection':
            paths, coco = contents.keys(), contents.values()
            paths = [os.path.join(
                self._dataset_root, 'images', i) for i in paths]
            contents = dict({k: v for k, v in zip(paths, coco)})

        # If the export format is `None`, we return the default mapping.
        if export_format is None:
            return contents

        # If the export format is `arrays`, return the keys and
        # the values of the mapping as two independent arrays.
        if export_format == 'arrays':
            return list(contents.keys()), list(contents.values())

        # A special case for COCO JSON dictionaries.
        if export_format == 'coco':
            if self._task != 'object_detection':
                raise ValueError("The `coco` export format is "
                                 "only for object detection tasks.")
            return self._default_coco_annotations

    # The following methods actually generate the content mappings for
    # the different tasks. In essence, for each image path, `image`, a
    # mapping is generated with a corresponding annotation, such as a
    # label for image classification, mask for semantic segmentation, or
    # a COCO JSON dictionary for object detection.

    def _generate_image_classification_data(self):
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
                image_label_mapping[file_] = self._info_map[dir_]
        self._data = image_label_mapping

    def _generate_image_regression_data(self):
        """Loads image regression data for the loader.

        In this format, there are input images in an `images` folder as
        well as other image formats in other various `*_images` folders,
        and an `annotations.json` file containing the regression outputs.
        """
        with open(os.path.join(self._dataset_root, 'annotations.json'), 'r') as f:
            annotations = json.load(f)
        content_mapping = {'inputs': [], 'outputs': []}
        annotation_types = set(list(self._info_map.keys()))
        annotation_types.remove('regression')
        for sample in annotations:
            for k, v in sample.items():
                if is_image_file(v):
                    sample[k] = os.path.join(self._dataset_root, f'{k}s', v)
            content_mapping['inputs'].append({
                k: v for k, v in sample.items()
                if re.match('(.*?)image', k)})
            out = {'regression': list(sample['outputs']['regression'].values())}
            out.update({
                k: self._info_map[k][v] for k, v in sample['outputs'].items()
                if k in annotation_types})
            content_mapping['outputs'].append(out)
        self._data = content_mapping

    def _generate_semantic_segmentation_data(self):
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
        self._data = image_annotation_map

    def _generate_object_detection_data(self):
        """Generates object detection data for the loader.

        Image data is loaded from an `images` directory, and the COCO
        JSON annotations are loaded from an `annotations.json` file.
        """
        with open(os.path.join(self._dataset_root, 'annotations.json')) as f:
            self._default_coco_annotations = json.load(f)
        coco_annotations = self._default_coco_annotations
        categories, labels = coco_annotations['categories'], []
        for category in categories:
            labels.append(category['name'])
        image_id_mapping = {}
        for img_meta in coco_annotations['images']:
            image_id_mapping[img_meta['id']] = img_meta['file_name']
        coco_map = {fname: [] for fname in image_id_mapping.values()}
        for a_meta in coco_annotations['annotations']:
            coco_map[image_id_mapping[a_meta['image_id']]].append(a_meta)
        self._data = coco_map

