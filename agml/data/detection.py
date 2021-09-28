import os
import json

import cv2
import numpy as np

from agml.backend.tftorch import set_backend, get_backend

from agml.utils.io import get_file_list
from agml.utils.general import resolve_list_value

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
        super(AgMLObjectDetectionDataLoader, self).__init__(dataset)

        # Process internal logic.
        self._shuffle = True
        if not kwargs.get('shuffle', True):
            self._shuffle = False

        # Build the data.
        self._find_images_and_annotations()

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
        meta_map = {fname: [] for fname in image_id_mapping.values()}
        class_map = {fname: None for fname in image_id_mapping.values()}
        for a_meta in self._coco_annotations['annotations']:
            meta_map[image_id_mapping[a_meta['image_id']]] \
                = a_meta['bbox']
            class_map[image_id_mapping[a_meta['image_id']]] \
                = a_meta['category_id'] - 1
        self._bbox_annotation_map = meta_map
        self._class_annotation_map = class_map

    def _reshuffle(self):
        """Reshuffles the data if allowed to."""
        if not self._shuffle:
            return
        bbox_items = list(self._bbox_annotation_map.items())
        class_items = list(self._class_annotation_map.items())
        per = np.random.permutation(len(bbox_items))
        self._bbox_annotation_map = dict(list(
            np.array(bbox_items, dtype = object)[per]))
        self._class_annotation_map = dict(list(
            np.array(class_items, dtype = object)[per]))

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
        A dictionary with the COCO JSON annotations.
        """
        return self._coco_annotations


