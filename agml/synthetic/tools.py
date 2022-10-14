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

"""Tools for working with synthetic data in the Helios format."""

import os

import math
from typing import List, Union, Tuple

import numpy as np
from scipy.signal import sawtooth

from agml.utils.io import get_dir_list


def generate_environment_map(
        origin: List[Union[int, float]],
        plant_spacing: int = 1,
        row_spacing: int = 4,
        trees_per_row: int = 1,
        rows: int = 1,
        plant_height: int = 1) -> list:
    """Generates an environment map in the Helios format.

    This method returns a list of NumPy arrays which contain coordinates
    of the canopies to be generated as specified by the provided parameters.
    These can be plotted/inspected to view the arrangement of the plants.

    Parameters
    ----------
    origin: {list, np,ndarray}
        A three-dimensional coordinate indicating the origin of the plot.
    plant_spacing : int
        The spacing of the plants.
    row_spacing : int
        The spacing between each row of plants.
    trees_per_row : int
        The number of trees to keep in each row of plants.
    rows : int
        The number of rows to generate.
    plant_height : 1
        The height of each plant generated.

    Returns
    -------
    A list of NumPy arrays representing each row, with the contents of each
    array being the coordinates of each plant.
    """
    # If there is only one row and only one tree in the row, then there
    # is only one plant whose coordinates needs to be generated.
    if trees_per_row == 1 and rows == 1:
        return [[np.concatenate([np.array([origin[0]]), np.array([origin[1]]),
                                 plant_height / 2 + np.array([origin[2]])])]]

    # Calculate the x- positions for the trees.
    if trees_per_row % 2 != 0 and trees_per_row > 1:  # odd number
        x_pos = np.concatenate(
            [np.array([origin[0]]), np.linspace(
                origin[0] + plant_spacing, trees_per_row / 2 * plant_spacing -
                plant_spacing / 2, int(trees_per_row / 2))])
        x_pos = np.concatenate([np.sort(-x_pos[1:]), x_pos])
    else:  # even number
        x_pos = np.linspace(origin[0] + (plant_spacing / 2),
                            trees_per_row / 2 * plant_spacing - plant_spacing / 2,
                            int(trees_per_row / 2))
        x_pos = np.concatenate([np.sort(-x_pos[:]), x_pos])

    # Calculate the y- positions for the trees.
    if rows % 2 != 0:  # odd number
        y_pos = np.concatenate(
            [np.array([origin[0]]), np.linspace(
                origin[1] + row_spacing, rows / 2 * row_spacing -
                row_spacing / 2, int(rows / 2))])
        y_pos = np.concatenate([np.sort(-y_pos[1:]), y_pos])
    else:
        y_pos = np.linspace(origin[1] + (row_spacing / 2),
                            rows / 2 * row_spacing - row_spacing / 2, int(rows / 2))
        y_pos = np.concatenate([np.sort(-y_pos[:]), y_pos])

    # Return the x- and y- positions.
    return [[[x_pos[x], y_pos[y], plant_height / 2 + origin[2]]
             for x in range(len(x_pos))] for y in range(len(y_pos))]


def generate_camera_positions(
        camera_type: str,
        num_views: int,
        origin: List[Union[int, float]] = None,
        camera_spacing: int = 2,
        crop_distance: int = 4,
        height: int = 1,
        aerial_parameters: dict = {}) -> Tuple:
    """Generates camera placement and lookat positions in the Helios format.

    This method, given the origin and a number of camera positions to generate
    (as specified in `num_views`), returns the 3-d coordinates of the camera
    positions as well as lookat vectors symbolizing their perspective of the
    provided origin point.

    Parameters
    ----------
    camera_type : str
        The type of camera position to generate: 'circular', 'linear',
        or 'aerial'. The lookat vectors are adjusted accordingly.
    num_views : int
        The number of different camera view positions to generate.
    origin : {list, np.ndarray}
        The origin point the cameras are focused on.
    camera_spacing : int
        The spacing of the cameras around a central point.
    crop_distance : int
        The distance of the cameras from the crop itself.
    height : int
        The height of the cameras relative to the crop.
    aerial_angled : bool
        Whether the aeiral camera should be angled or directly facing down.

    Returns
    -------
    A tuple of two values: the camera positions and the lookat vectors.
    """
    if origin is None:
        origin = [0, 0, 0]

    if camera_type == 'circular':
        return [[math.cos(2 * math.pi / num_views * x) * crop_distance, 
                  math.sin(2 * math.pi / num_views * x) * crop_distance, height] 
                 for x in range(0, num_views)], [[0, 0, 1] for _ in range(0, num_views)]

    elif camera_type == 'linear':
        camera_pos = np.arange(
            origin[0], camera_spacing * num_views + origin[0], camera_spacing)
        return [[camera_pos[x], crop_distance + origin[1], height]
                for x in range(len(camera_pos))], \
                [[camera_pos[x], origin[0], height] for x in range(len(camera_pos))]

    elif camera_type == 'aerial':
        if aerial_parameters.get('distribution', '') == 'sawtooth':
            angled = aerial_parameters.get('angled', False)
            t = camera_spacing * np.linspace(0, 1, num_views)
            triangle = camera_spacing * sawtooth(1 * np.pi * 5 * t, 0.5)
            return [[t[x] + origin[0], triangle[x] + origin[1],
                      crop_distance + origin[2]] for x in range(len(triangle))], \
                    [[t[x] + origin[0], triangle[x] + origin[1] + (1 if angled else 0.05),
                      height + origin[2]] for x in range(len(triangle))]
        else: # distribution is circular around the center
            angled = aerial_parameters.get('angled', False)
            center_coord = [[origin[0], origin[1]]]
            if num_views == 1:
                coords = center_coord
            elif num_views == 2:
                coords = [center_coord[0], [origin[0], origin[1] + 0.5]]
            elif num_views == 3:
                coords = [center_coord[0], [origin[0], origin[1] - 0.5],
                          [origin[0], origin[1] + 0.5]]
            else:
                coords = center_coord
                coords.extend(
                    [[origin[0] + np.cos(theta), origin[1] + np.sin(theta)]
                     for theta in np.linspace(0, 2 * np.pi, num_views)])

            return [[*coord, crop_distance + origin[2]] for coord in coords], \
                   [[coord[0], coord[1] + (1 if angled else 0.05),
                     height + origin[2]] for coord in coords]

    else:
        raise ValueError(f"Got `camera_type`: ({camera_type}), "
                         f"expected either `circular`, `linear`, or `aerial`.")


def _is_agml_converted(dataset_path):
    """Returns whether a Helios dataset has been converted to AgML format."""
    return os.path.exists(os.path.join(dataset_path, '.metadata', 'agml_info.json')) \
            or get_dir_list(dataset_path) == ['.metadata'] # dataset with no annotations
