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

import cv2
import numpy as np

from agml.framework import AgMLSerializable
from agml.data.loader import AgMLDataLoader
from agml.data.public import public_data_sources
from agml.utils.io import nested_file_list
from agml.utils.random import inject_random_state


class ImageLoader(AgMLSerializable):
    """A data loader designed purely for images, with no annotations.

    This is a bare-bones data loader designed for loading images, which is useful
    when prototyping new models or when you have a dataset with no annotations.
    It involves minimal functionality (at least as of now), with a few options for
    modifying the input image, such as resizing and converting to grayscale.

    Parameters
    ----------
    location : str
        The location of the image files. This can be either a directory of images,
        a directory tree with nested image folders, or an AgML dataset.
    image_size : tuple, optional
        The size to resize the images to. If `None`, the images will not be resized.
    """
    serializable = frozenset(
        ('accessor_list', 'image_size', 'grayscale', 'return_paths', 'transforms'))

    def __init__(self, location, image_size = None, **kwargs):
        # Parse the input `location`: this can be either a directory of images,
        # a directory tree with nested image folders, or an AgML dataset.
        self._setup_loader(location, **kwargs)

        # Store an accessor list which can be shuffled.
        self._accessor_list = np.array(list(range(len(self._image_files))))

        # Save class variables.
        self._image_size = None
        self.image_size = image_size # checks for validity.
        self._grayscale = False
        self._return_paths = False
        self._transforms = []

    def __getitem__(self, index):
        # Load the image and convert it to RGB format.
        accessor_value = self._accessor_list[index]
        image_file = self._image_files[accessor_value]
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)

        # Apply any transforms (this is conducted before resizing and grayscale
        # conversion in order to retain as much information as possible).
        for transform in self._transforms:
            image = transform(image)

        # Perform necessary input operations based on the parameters.
        if self._image_size is not None:
            image = cv2.resize(image, self._image_size, cv2.INTER_LINEAR)
        if self._grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Return the input image (and potentially the path).
        if self._return_paths:
            return image, image_file
        return image

    def __len__(self):
        return len(self._accessor_list)

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, value):
        if value is not None:
            if isinstance(value, int):
                value = (value, value)
            if len(value) != 2:
                raise ValueError("Cannot use `image_size` with more than two dimensions.")
            value = (value[0], value[1]) # convert to tuples for cv2.resize
        self._image_size = value

    @property
    def grayscale(self):
        return self._grayscale

    @grayscale.setter
    def grayscale(self, value):
        self._grayscale = value

    @property
    def return_paths(self):
        return self._return_paths

    @return_paths.setter
    def return_paths(self, value):
        self._return_paths = value

    @inject_random_state
    def shuffle(self):
        """Shuffles the order of the images in the dataset."""
        np.random.shuffle(self._accessor_list)

    def _setup_loader(self, location, **kwargs):
        # If the input is an AgML dataset, then all the images will be either
        # contained within a single folder `images`, or in sub-directories which
        # are named after the class (for image classification datasets only).
        if location in public_data_sources():
            # This is a quick way to run all of the checks regarding the location
            # of the dataset, and download it if it isn't already downloaded.
            self._root_path = AgMLDataLoader(location, **kwargs).dataset_root
            self._image_files = sorted(nested_file_list(self._root_path))

        # If the folder itself is a directory of images, then we can just use
        # the directory as the root path.
        elif os.path.isdir(os.path.abspath(location)):
            self._root_path = os.path.abspath(location)

            # Get all of the image files.
            self._image_files = sorted(nested_file_list(self._root_path))

        # Otherwise, the given path is invalid.
        else:
            raise ValueError(f'Invalid path for `ImageLoader`: {location}')

    def transform(self, transform):
        """Adds a transform to the loader.

        Parameters
        ----------
        transform : callable
            A function which takes in an image and returns a modified image.
        """
        self._transforms.append(transform)

