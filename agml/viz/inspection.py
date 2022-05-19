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
import glob

import cv2

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from agml.viz.tools import show_when_allowed
from agml.viz.labels import _inference_best_shape
from agml.synthetic.tools import _is_agml_converted
from agml.utils.image import imread_context


class Arrow3D(FancyArrowPatch):
    """Draws a 3-dimensional arrow using the given coordinates."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._coords = xs, ys, zs

    def draw(self, renderer):
        xs, ys, zs = self._coords
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


@show_when_allowed
def plot_synthetic_camera_positions(positions, lookat):
    """Plots camera perspectives: positions and lookat vectors.

    This method is used to plot camera perspectives from a provided
    set of positional coordinates and lookat vector coordinates. This
    can be used to visualize the perspectives of cameras when generating
    synthetic agricultural data using Helios.

    The output of the method `agml.synthetic.generate_camera_positions`
    can be piped directly into this method to get a plot straight from
    environment parameters.

    Parameters
    ----------
    positions : list
        A list of 3-d coordinates indicating the positions of the
        cameras whose perspectives are to be plotted.
    lookat : list
        A list of lookat vector coordinates.

    Returns
    -------
    The matplotlib figure with the plot.
    """
    # Creat ethe figure.
    plt.figure(figsize = (7, 7))
    ax = plt.axes(projection = '3d')
    ax.plot3D(0, 0, 1, 'r*', label = 'Canopy Origin')
    for i in range(len(positions)):
        ax.plot3D(positions[i][0], positions[i][1], positions[i][2], 'o',
                  label = 'Camera ' + str(i))
        arw = Arrow3D([positions[i][0], lookat[i][0]],
                      [positions[i][1], lookat[i][1]],
                      [positions[i][2], lookat[i][2]],
                      arrowstyle = "->", color = "purple",
                      lw = 1, mutation_scale = 25)
        ax.add_artist(arw)
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.05),
              ncol = 4, fancybox = True, shadow = True)
    fig = plt.gcf()
    return fig


@show_when_allowed
def visualize_all_views(dataset_path, image):
    """Plots all of the camera views for a specific rendered canopy.

    Given a path to a set of camera views for a rendered canopy, this method
    will generate a grid onto which all of the views will be displayed. This
    enables inspection of a plant from multiple angles, supposing that there
    are multiple views generated for it.

    Note that the dataset can either be in the original Helios-generated format,
    or it can be converted. This method will either way find all of the views
    for the provided input image number.

    Parameters
    ----------
    dataset_path : str
        The path to the entire dataset.
    image : {int, str}
        An integer (possibly in string format) representing the number of the
        image whose camera views you want to view.

    Returns
    -------
    The matplotlib figure with the plot.
    """
    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
    if not os.path.exists(dataset_path):
        raise NotADirectoryError(
            f"The provided dataset path {dataset_path} does not exist.")

    # Get all of the views for the input image.
    image = str(image)
    if image.startswith('image'):
        image = image.replace('image', '')
    if _is_agml_converted(dataset_path):
        views = glob.glob(os.path.join(
            dataset_path, 'images', f'image{image}-view*.jpeg'))
        if len(views) == 0:
            views = glob.glob(os.path.join(
                dataset_path, f'image{image}-view*.jpeg'))
    else:
        views = glob.glob(os.path.join(
            dataset_path, f'image{image}', '**/*.jpeg'
        ), recursive = True)

    # Check that there are images.
    if len(views) == 0:
        raise FileNotFoundError(f"Could not find any views for image '{image}'.")

    # Parse all of the images.
    images = []
    for view in views:
        with imread_context(view) as img:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Construct the figure.
    shape = _inference_best_shape(len(images))
    fig, axes = plt.subplots(shape[0], shape[1], figsize = (shape[1] * 5, shape[0] * 5))
    for image, ax in zip(images, axes.flat):
        ax.imshow(image)
        ax.set_axis_off()

    # Return the figure.
    fig.tight_layout()
    return fig



