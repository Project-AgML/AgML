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

from agml.viz.tools import auto_resolve_image, show_when_allowed, format_image
from agml.backend.tftorch import as_scalar, is_array_like


def _inference_best_shape(n_images):
    """Inferences the best matplotlib row/column layout.

    This method searches for the two closest factors of the number
    `n_images`, and returns this tuple as the best shape, since this
    is the closest to a square grid as possible.
    """
    a, b, i = 1, n_images, 0
    while a < b:
        i += 1
        if n_images % i == 0:
            a = i
            b = n_images // a
    return [b, a]


@show_when_allowed
@auto_resolve_image
def visualize_images(images, shape = None):
    """Visualizes a set of images in the given shape.

    Given a set of images, this method will generate a grid for the
    images and display them as such. The shape of the grid will by
    default be inferenced to be the two closest factors of the number
    of images (to  be as close to square as possible).

    Parameters
    ----------
    images : Any
        Either a list of images, a tuple of images and labels, or a list
        of image/label pairs (like you would get as the output of a dataset).
    shape : Any
        The shape of the display grid.

    Returns
    -------
    The matplotlib figure with the plotted images.
    """
    # If a prime number is passed, e.g. 23, then the `_inference_best_shape`
    # method will return the shape of (23, 1). Likely, the user is expecting
    # a non-rectangular shape such as (6, 4), where the bottom right axis is
    # empty. This method does not support such computations (yet).
    if shape is None:
        shape = _inference_best_shape(len(images))
    if max(shape) > 20:
        raise NotImplementedError(
            "Length of maximum shape length is greater than 20. "
            "This method does not support non-rectangular shapes.")

    fig, axes = plt.subplots(
        shape[0], shape[1], figsize = (shape[1] * 2, shape[0] * 2))
    try:
        iter_ax = axes.flat
    except AttributeError: # If showing only a single image.
        iter_ax = [axes]
    for image, ax in zip(images, iter_ax):
        ax.imshow(format_image(image))
        ax.set_aspect(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(
            axis = 'both', which = 'both', bottom = False,
            top = False, left = False, right = False
        )
        plt.setp(ax.spines.values(), visible = False)

    fig.tight_layout()
    return fig



@show_when_allowed
@auto_resolve_image
def visualize_images_with_labels(images, labels = None, *, info = None, shape = None):
    """Visualizes a set of images with their classification labels.

    Given a set of images and their corresponding labels, this method
    will generate a grid for the images and display them with their
    image classification labels displayed underneath them. The shape of
    the grid will by default be inferenced to be the two closest factors
    of the number of images (to be as close to square as possible).

    If you provide an `info` parameter, which will consist of the `info`
    property of an AgMLDataLoader (literally pass `loader.info`), then the
    method will convert the classification numbers to their label names.

    Parameters
    ----------
    images : Any
        Either a list of images, a tuple of images and labels, or a list
        of image/label pairs (like you would get as the output of a dataset).
    labels : Any
        A list or array of classification labels.
    info : DatasetMetadata
        The `loader.info` attribute of a dataloader.
    shape : Any
        The shape of the display grid.

    Returns
    -------
    The matplotlib figure with the plotted info.
    """
    if images is not None and labels is None:
        if is_array_like(images[0]):
            if images[0].ndim >= 3:
                images, labels = images[0], images[1]
            else:
                raise ValueError(
                    "If passing a numpy array for images, expected at "
                    "least three dimensions: (batch, height, width).")
        elif isinstance(images[0], (tuple, list)):
            if isinstance(images[0][0], np.ndarray):
                if len(images[0]) == 2:
                    _images, _labels = [], []
                    for content in images:
                        _images.append(content[0])
                        _labels.append(content[1])
                    images, labels = _images, _labels
                else:
                    images, labels = images[0], images[1]
    if labels is None:
        raise TypeError(
            "Invalid format for `images` and `labels`, see documentation.")
    if isinstance(images, np.ndarray) and images.shape[0] > 100:
        images, labels = [images], [labels]

    # Check if the labels are converted to one-hot, and re-convert them back.
    if is_array_like(labels):
        if labels.ndim == 2: # noqa
            labels = np.argmax(labels, axis = -1)

    # If a prime number is passed, e.g. 23, then the `_inference_best_shape`
    # method will return the shape of (23, 1). Likely, the user is expecting
    # a non-rectangular shape such as (6, 4), where the bottom right axis is
    # empty. This method does not support such computations (yet).
    if shape is None:
        shape = _inference_best_shape(len(images))
    if max(shape) > 20:
        raise NotImplementedError(
            "Length of maximum shape length is greater than 20. "
            "This method does not support non-rectangular shapes.")

    fig, axes = plt.subplots(
        shape[0], shape[1], figsize = (shape[1] * 2, shape[0] * 2))
    try:
        iter_ax = axes.flat
    except AttributeError: # If showing only a single image.
        iter_ax = [axes]
    for image, label, ax in zip(images, labels, iter_ax):
        ax.imshow(format_image(image))
        ax.set_aspect(1)
        label = as_scalar(label)
        if info is not None:
            label = info.num_to_class[label]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(
            axis = 'both', which = 'both', bottom = False,
            top = False, left = False, right = False
        )
        plt.setp(ax.spines.values(), visible = False)
        ax.set_xlabel(label)

    fig.tight_layout()
    return fig



