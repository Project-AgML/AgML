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

import matplotlib.pyplot as plt

from agml.backend.tftorch import is_array_like
from agml.viz.boxes import show_image_and_boxes
from agml.viz.masks import show_image_and_overlaid_mask
from agml.viz.labels import show_images_and_labels
from agml.viz.tools import format_image, _inference_best_shape, convert_figure_to_image
from agml.viz.display import display_image
import numpy as np
import random

import numpy as np
import random

import numpy as np
import random

#   Works best - for classes starting from 1
def show_sample(loader, image_only=False, num_images=1, **kwargs):
    """A simplified convenience method that visualizes multiple samples from a loader.

    This method works for all types of annotations and visualizes multiple
    images based on the specified `num_images` parameter.

    Parameters
    ----------
    loader : AgMLDataLoader
        An AgMLDataLoader of any annotation type.
    image_only : bool
        Whether to only display the images.
    num_images : int
        The number of images to display (default is 1).

    Returns
    -------
    The matplotlib figure showing the requested number of images.
    """
    if kwargs.get('sample', None) is not None:
        sample = kwargs['sample']
    else:
        sample = loader[0]

    if image_only:
        return show_images(sample[0])

    # if loader.task == 'object_detection':
    #     classes = loader.classes
    #     num_classes = len(classes)
    #     print("Number of classes", num_classes)
    #     return show_image_and_boxes(
    #         sample, info = loader.info, no_show = kwargs.get('no_show', False))

    if loader.task == 'object_detection':
    # Collect 4 images (or as many as available if fewer)
        samples = [loader[i] for i in range(min(len(loader), 4))]
        return [show_image_and_boxes(sample, info=loader.info, no_show=kwargs.get('no_show', False)) for sample in samples]

    elif loader.task == 'semantic_segmentation':
        return show_image_and_overlaid_mask(
            sample, no_show = kwargs.get('no_show', False))
    elif loader.task == 'image_classification':
            # Fetch all available classes and initialize an empty list for samples.
        classes = loader.classes
        num_classes = len(classes)
        samples = []

        # Adjust the dictionary to store samples per class, starting from class 1.
        class_to_sample = {cls: None for cls in range(num_classes)}

        # Ensure one image from each class first (without duplication).
        for i in range(len(loader)):
            sample = loader[i]
            label = sample[1]

            # Map label to its class index (if necessary)
            if isinstance(label, (list, np.ndarray)):  # Handle one-hot encoding case
                label = np.argmax(label) # Adjust indexing to start from 1
            
            if isinstance(label, dict):
                label = label.get('label', None) 

            if label in class_to_sample and class_to_sample[label] is None:
                class_to_sample[label] = sample

            # Stop once we have at least one image per class
            if all(v is not None for v in class_to_sample.values()):
                break

        # Collect samples ensuring uniqueness per class until we hit num_classes.
        samples = [class_to_sample[cls] for cls in range(num_classes) if class_to_sample[cls] is not None]

        # If more images are required, duplicate randomly from the collected samples.
        if num_images > num_classes:
            additional_samples = random.choices(samples, k=num_images - num_classes)
            samples.extend(additional_samples)
        
        # If fewer images are requested, truncate the sample list.
        if num_images <= num_classes:
            samples = samples[:num_images]
        return show_images_and_labels(
            samples, info=loader.info, no_show=kwargs.get('no_show', False))


# Working fine
# def show_sample(loader, image_only=False, num_images=1, **kwargs):
#     """A simplified convenience method that visualizes multiple samples from a loader.

#     This method works for all types of annotations and visualizes multiple
#     images based on the specified `num_images` parameter.

#     Parameters
#     ----------
#     loader : AgMLDataLoader
#         An AgMLDataLoader of any annotation type.
#     image_only : bool
#         Whether to only display the images.
#     num_images : int
#         The number of images to display (default is 1).

#     Returns
#     -------
#     The matplotlib figure showing the requested number of images.
#     """
#     # Fetch the requested number of samples from the loader.
#     samples = [loader[i] for i in range(num_images)]

#     # Visualize the images based on the task.
#     if image_only:
#         return show_images([sample[0] for sample in samples])

#     if loader.task == 'object_detection':
#         return [show_image_and_boxes(
#             sample, info=loader.info, no_show=kwargs.get('no_show', False))
#             for sample in samples]
#     elif loader.task == 'semantic_segmentation':
#         return [show_image_and_overlaid_mask(
#             sample, no_show=kwargs.get('no_show', False))
#             for sample in samples]
#     elif loader.task == 'image_classification':
#         return show_images_and_labels(
#             samples, info=loader.info, no_show=kwargs.get('no_show', False))


# def show_sample(loader, image_only = False, **kwargs):
#     """A simplified convenience method that visualizes a sample from a loader.

#     This method works for all kind of annotations; it picks the appropriate
#     visualization method and then calls it with a sample from the loader.
#     If you want to customize the way the output looks, then you need to use
#     the actual methods directly.

#     Parameters
#     ----------
#     loader : AgMLDataLoader
#         An AgMLDataLoader of any annotation type.
#     image_only : bool
#         Whether to only display the image.

#     Returns
#     -------
#     The matplotlib figure.
#     """
#     if kwargs.get('sample', None) is not None:
#         sample = kwargs['sample']
#     else:
#         sample = loader[0]

#     if image_only:
#         return show_images(sample[0])

#     if loader.task == 'object_detection':
#         return show_image_and_boxes(
#             sample, info = loader.info, no_show = kwargs.get('no_show', False))
#     elif loader.task == 'semantic_segmentation':
#         return show_image_and_overlaid_mask(
#             sample, no_show = kwargs.get('no_show', False))
#     elif loader.task == 'image_classification':
#         return show_images_and_labels(
#             sample, info = loader.info, no_show = kwargs.get('no_show', False))


def show_images(images,
                shape = None,
                **kwargs):
    """Shows multiple images in a grid format with the given shape.

    Given a set of images, this method will generate a grid for the
    images and display them as such. The shape of the grid will by
    default be inferenced to be the two closest factors of the number
    of images (to be as close to square as possible).

    If you don't want to display the image (and just get the output), pass
    `no_show` as true in order to bypass this.

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
    # If only one image is passed, then we can directly display that image.
    if not isinstance(images, (list, tuple)):
        if is_array_like(images):
            images = format_image(images)
            if not kwargs.get('no_show', False):
                display_image(images)
            return images

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

    # Display and return the image.
    image = convert_figure_to_image()
    if not kwargs.get('no_show', False):
        display_image(image)
    return image

