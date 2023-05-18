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

import numpy as np

import matplotlib.pyplot as plt

from agml.data.point_cloud import PointCloud
from agml.utils.logging import log


def show_point_cloud(point_cloud, format = 'default'):
    """Visualizes a point cloud in 3-dimensional space.

    Parameters
    ----------
    point_cloud : PointCloud
        The point cloud to visualize.
    format : str
        The format to use for visualization. If 'default', then the
        visualization will be done using Open3D if it is installed, and
        matplotlib otherwise. If 'open3d', then the visualization will be
        done using Open3D. If 'matplotlib', then the visualization will be
        done using matplotlib. Defaults to 'default'.
    """
    try:
        import open3d as o3d
    except:
        log("Open3D is not installed. If you want to have high-res point cloud "
            "visualizations, please install Open3D using `pip install open3d`. "
            "For now, defaulting to using 3D matplotlib visualizations.")
        o3d = None

    # We need to hard-code this check because this method isn't designed to
    # run with regular point cloud arrays, only `point_cloud` objects.
    if not isinstance(point_cloud, PointCloud):
        raise TypeError(f'`point_cloud` must be a `PointCloud` object, instead got {point_cloud}.')

    # If Open3D is installed, we can just use that.
    if o3d is not None and format in ['open3d', 'default']:
        o3d.visualization.draw_geometries([point_cloud.structure_3d])
        return

    # Otherwise, use matplotlib to plot the points.
    coords = point_cloud._coordinates
    x, y, z = np.rollaxis(coords, axis = 1)
    fig = plt.figure(figsize = (16, 16))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, c = point_cloud.labels)
    plt.show()




