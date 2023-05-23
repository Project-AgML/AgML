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
import glob

import numpy as np

from agml.framework import AgMLSerializable
from agml.backend.config import synthetic_data_save_path
from agml.data.point_cloud import PointCloud
from agml.utils.random import inject_random_state


class LiDARDataLoader(AgMLSerializable):
    """A data loader for Helios-generated LiDAR point clouds."""
    serializable = frozenset(('point_clouds', 'accessor_list'))

    def __init__(self, location):
        # Parse the input location and load all of the point clouds.
        self._setup_loader(location)

        # Store an accessor list which can be shuffled.
        self._accessor_list = np.array(list(range(len(self._point_clouds))))

    def __len__(self):
        return len(self._point_clouds)

    def __getitem__(self, index):
        return self._point_clouds[index]

    @inject_random_state
    def shuffle(self):
        """Shuffle the order of the point clouds."""
        np.random.shuffle(self._accessor_list)

    def _setup_loader(self, location):
        # Check whether the path to the dataset exists (this should either be a
        # specific directory with point clouds or a dataset generated directly
        # from Helios). Datasets generated with Helios will have a stylesheet
        # XML from which we can directly load the point cloud structure.
        if os.path.isdir(location):
            pass
        else:
            location = os.path.join(synthetic_data_save_path(), location)
            if not os.path.isdir(location):
                raise ValueError(f'Invalid location: {location}')
        structure = self._load_structure(location)
        point_cloud_files = glob.glob(os.path.join(location, '**/*.xyz'), recursive = True)
        self._point_clouds = [PointCloud(f, structure = structure) for f in point_cloud_files]

    @staticmethod
    def _load_structure(location):
        # Check if the metadata directory exists, and if the stylesheet exists.
        style_query_path = os.path.join(location, '.metadata', 'style*.xml')
        if not os.path.exists(os.path.dirname(style_query_path)):
            return None
        style_file = glob.glob(style_query_path)
        if len(style_file) == 0:
            return None
        style_file = style_file[0]

        # Load the `ASCII_format` XML tag if it exists.
        contents = open(style_file, 'r').read()
        ascii_format = re.search('<ASCII_format>(.*)</ASCII_format>', contents).group(1).strip()
        ascii_format = ascii_format.replace('object_label', 'label')
        ascii_format = ascii_format.split(' ')
        return ascii_format

    def show_sample(self, format = 'default'):
        """Shows a sample of a point cloud in the loader."""
        # Prevent circular imports.
        from agml.viz.point_clouds import show_point_cloud
        show_point_cloud(self._point_clouds[np.random.choice(self._accessor_list)], format = format)


