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

import numpy as np

from matplotlib.pyplot import cm

from agml.utils.general import is_float, is_int
from agml.utils.logging import log


class PointCloud(object):
    """Represents a 3D point cloud object, with utilities for format + visualization."""

    # Valid structures for the point cloud file.
    VALID_STRUCTURES = {'x': float,
                        'y': float,
                        'z': float,
                        'label': int,
                        'intensity': float,
                        'r': int,
                        'g': float,
                        'b': float}

    # Mapping between string labels and their integer representation.
    LABEL_MAPPING = {'none': -9999,
                     'ground': 1,
                     'trunk': 2,
                     'shoot': 3,
                     'leaf': 4,
                     'fruit': 5}

    def __init__(self, contents, structure = None, colormap = None):
        # `contents` is expected to be the path to a point cloud `xyz` file, but can
        # also be the contents of the file itself, in which case there is not saved path.
        if isinstance(contents, str):
            contents = os.path.abspath(contents)
            if not os.path.exists(contents):
                raise FileNotFoundError(f'Point cloud file {contents} does not exist.')
            raw_contents = [i for i in open(contents, 'r').read().splitlines() if not i == '']
            self._raw_contents = raw_contents
            self._path = contents
        else:
            raw_contents = contents
            self._raw_contents = raw_contents
            self._path = None

        # Set the colormap for the point cloud visualization.
        self._colormap = colormap if colormap is not None else cm.rainbow(np.linspace(0, 1, 6))

        # Load the point cloud and the colors.
        self._structure = self._infer_structure(raw_contents[0], structure)
        self._read_point_cloud(raw_contents)

        # Build the 3-D point cloud.
        self._build_3d_object()

    def __len__(self):
        return len(self._coordinates)

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def labels(self):
        return self._labels

    @property
    def shape(self):
        return self._coordinates.shape

    @property
    def structure_3d(self):
        return self._structure_3d

    @staticmethod
    def _infer_structure(row, pointcloud_structure):
        """Infers the structure of the point cloud file from the first row."""
        if pointcloud_structure is not None:
            # If a string is passed, it should contain space/comma separated values.
            if isinstance(pointcloud_structure, str):
                if ',' in pointcloud_structure:
                    pointcloud_structure = [i.strip() for i in pointcloud_structure.split(',')]
                else:
                    pointcloud_structure = pointcloud_structure.split(' ')

            # If a list is passed, it should contain only strings.
            elif isinstance(pointcloud_structure, list):
                if not all(isinstance(i, str) for i in pointcloud_structure):
                    raise TypeError('Point cloud structure must be a list of strings.')

            # Check that each of the structural values is valid.
            if not all(i in PointCloud.VALID_STRUCTURES for i in pointcloud_structure):
                raise ValueError(f'Invalid point cloud structure {pointcloud_structure}, '
                                 f'should only contain values in {PointCloud.VALID_STRUCTURES.keys()}.')

        # If no structure is passed, check if it fits one of the common formats.
        else:
            # Preprocess the row.
            row = row.strip().split(' ')

            # Three float values is `x y z` format.
            if len(row) == 3:
                if all(is_float(i) for i in row):
                    pointcloud_structure = ['x', 'y', 'z']

            # Three float values and one integer value is `x y z label` format.
            elif len(row) == 4:
                if all(is_float(i) for i in row[:-1]) and is_int(row[-1]):
                    pointcloud_structure = ['x', 'y', 'z', 'label']

            # Eight values (three float, an integer, and four floats) is the
            # complete `x y z label intensity r g b` format.
            elif len(row) == 8:
                if (all(is_float(i) for i in row[:-4])
                        and all(is_int(i) for i in row[3:-1])
                        and all(is_float(i) for i in row[-4:])):
                    pointcloud_structure = ['x', 'y', 'z', 'label', 'intensity', 'r', 'g', 'b']

            # Otherwise, we have encountered some form of format that we do not support.
            else:
                raise ValueError('Could not infer point cloud structure from file.')

        # Check that the point cloud structure makes sense (e.g., at minimum it
        # should have `x y z`, and it shouldn't just have `r` and not `g b`, etc).
        if any(i not in pointcloud_structure for i in ['x', 'y', 'z']):
            raise ValueError('Point cloud structure must contain `x y z`.')
        if any(i in pointcloud_structure for i in ['r', 'g', 'b']) \
                and not all(i in pointcloud_structure for i in ['r', 'g', 'b']):
            raise ValueError('Point cloud structure must contain all of `r g b`.')

        return pointcloud_structure

    @staticmethod
    def _read_object_label(value):
        return PointCloud.LABEL_MAPPING[int(value)]

    def _read_point_cloud(self, contents):
        """Reads the contents of the point cloud file with the given structure."""
        full_data = []
        for line in contents:
            line = line.strip().split(' ')
            current_line = [PointCloud.VALID_STRUCTURES[fmt](i)
                            for i, fmt in zip(line, self._structure)]
            current_line = [i if i != -9999 else 0 for i in current_line] # fix this label
            full_data.append(current_line)

        # Create arrays with the data.
        xyz_indices = [self._structure.index(c) for c in ['x', 'y', 'z']]
        self._coordinates = np.array(full_data)[:, xyz_indices]
        self._full_data = np.array(full_data)

        # Load the corresponding colormap for the coordinates if possible.
        if 'label' in self._structure:
            self._labels = labels = self._full_data[:, self._structure.index('label')].astype(int)
        else:
            labels = np.zeros(shape = (len(full_data),), dtype = int)
            self._labels = None
        self._pointcloud_colors = [self._colormap[i][:3] for i in labels]

    def _build_3d_object(self):
        """Function to construct the point cloud as a open3d point cloud object
        """
        try:
            import open3d as o3d
        except:
            log("Open3D is not installed. If you want to have high-res point cloud "
                "visualizations, please install Open3D using `pip install open3d`. "
                "For now, defaulting to using 3D matplotlib visualizations.")
            o3d = None

        # Construct the point cloud with the points and colors.
        if o3d is not None:
            self._structure_3d = o3d.geometry.PointCloud()
            points, colors = o3d.utility.Vector3dVector(), o3d.utility.Vector3dVector()
            for color, coord in zip(self._pointcloud_colors, self._coordinates):
                points.append(coord)
                colors.append(color)
            self._structure_3d.points = points
            self._structure_3d.colors = colors
        else:
            self._structure_3d = None

    def show(self, format = 'default'):
        """Show a point cloud sample with its labels."""
        # Prevent circular imports.
        from agml.viz.point_clouds import show_point_cloud
        show_point_cloud(self, format = format)

