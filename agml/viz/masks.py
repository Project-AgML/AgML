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

import cv2
import numpy as np
import matplotlib.pyplot as plt

from agml.utils.general import resolve_tuple_values
from agml.viz.tools import get_colormap, format_image, convert_figure_to_image
from agml.viz.display import display_image


def binary_to_channel_by_channel(mask, num_classes = None):
    """Converts a mask from 2-dimensional to channel-by-channel."""
    # If the mask is already 3 channels (in channel-by-channel format),
    # then don't do anything and return the original mask as-is.
    if mask.ndim == 3 and not np.all(mask[:, :, 0] == mask[:, :, 1]):
        return mask

    # For binary classification tasks, return the mask with an additional channel.
    if len(np.unique(mask)) == 2 and mask.max() == 1:
        return np.expand_dims(mask, -1)

    # Otherwise, convert the mask to channel-by-channel format.
    input_shape = mask.shape
    mask = mask.ravel()
    n = mask.shape[0]
    mask = np.array(mask, dtype = np.int32)
    if num_classes is None:
        num_classes = np.max(mask) + 1
    out = np.zeros(shape = (n, num_classes))
    out[np.arange(n), mask] = 1
    out = np.reshape(out, input_shape + (num_classes,))
    return out[..., 1:].astype(np.int32)


def convert_mask_to_colored_image(mask):
    """Converts a semantic segmentation mask into a colored image.

    Parameters
    ----------
    mask : np.ndarray
        A semantic segmentation mask.

    Returns
    -------
    The post-processed mask with different colors for each class.
    """
    # Check that the mask has an appropriate number of dimensions. If it is
    # 2-dimensional, then convert it into a three-dimensional mask. Otherwise,
    # if all of the channel slices are the same, then it is fine. If not, then
    # convert it from a channel-by-channel mask to a 2-dimensional mask.
    if mask.ndim == 3:
        if not np.all(mask[:, :, 0] == mask[:, :, 1]):
            mask = np.argmax(mask, axis = -1)
    if mask.ndim == 2:
        mask = np.dstack((mask, mask, mask))

    # Add a colorscheme to the labels.
    cmap = get_colormap()
    label_indices = np.unique(mask)[1:] # skip the background label
    for label_indx in label_indices:
        x_values, y_values, *_ = np.where(mask == label_indx)
        mask[tuple([*np.stack([x_values, y_values])])] = cmap[label_indx]
    return mask


def annotate_semantic_segmentation(image,
                                   mask = None,
                                   alpha = 0.3,
                                   border = True):
    """Annotates a semantic segmentation mask over an image.

    This method overlays a segmentation mask over an image. It uses contours
    to draw out the segmentation borders. The mask can be either a binary mask
    or a channel-by-channel mask. The level to which the overlay is done can be
    controlled using the `alpha` argument, and if you want to see the actual
    borders of each of the individual segmented blobs, then set `border` to True.

    Parameters
    ----------
    image : np.ndarray
        The image to annotate.
    mask : np.ndarray, optional
        The segmentation mask to overlay over the image.
    alpha : float, optional
        The level of transparency to use when overlaying the mask over the image.
    border : bool, optional
        Whether or not to draw the actual borders of the segmented blobs.

    Returns
    -------
    The annotated image.
    """
    # Get the image and mask from the given input arguments.
    image, mask = resolve_tuple_values(
        image, mask, custom_error =
        "If `image` is a tuple/list, it should contain "
        "two values: the image and its mask.")
    image = format_image(image)
    mask = binary_to_channel_by_channel(format_image(mask, mask = True))

    # Plot the given segmentation mask over the image. This essentially
    # overlays the given mask using contours to draw out segmentation borders.
    cmap = get_colormap()
    mask = np.transpose(mask, (2, 0, 1))
    for level, label in enumerate(mask):
        label = np.expand_dims(label, axis = -1)
        label = (label * np.true_divide(255, np.max(label, axis = (0, 1)) + 1e-6)).astype(np.uint8)

        # Find the given contours and plot them.
        _, thresh = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for indx, contour in enumerate(contours):
            color = cmap[level + 1]
            overlay = image.copy()
            cv2.fillPoly(overlay, pts = [contour], color = color)
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Draws the actual contour borders to show the segmented blobs.
            if border:
                image = cv2.polylines(
                    image, pts = [contour], isClosed = True,
                    color = color, thickness = 2)
    return image


def show_image_and_overlaid_mask(image,
                                 mask = None,
                                 alpha = 0.3,
                                 border = True,
                                 **kwargs):
    """Displays an image with an annotated segmentation mask.

    This method overlays a segmentation mask over an image. It uses contours
    to draw out the segmentation borders. The mask can be either a binary mask
    or a channel-by-channel mask. The level to which the overlay is done can be
    controlled using the `alpha` argument, and if you want to see the actual
    borders of each of the individual segmented blobs, then set `border` to True.

    Parameters
    ----------
    image : np.ndarray
        The image to annotate.
    mask : np.ndarray, optional
        The segmentation mask to overlay over the image.
    alpha : float, optional
        The level of transparency to use when overlaying the mask over the image.
    border : bool, optional
        Whether or not to draw the actual borders of the segmented blobs.

    Returns
    -------
    The annotated image.
    """
    # Parse the inputs and annotate the image.
    image = annotate_semantic_segmentation(image = image,
                                           mask = mask,
                                           alpha = alpha,
                                           border = border)

    # Display the annotated image.
    if not kwargs.get('no_show', False):
        _ = display_image(image, matplotlib_figure = False)
    return image


def show_image_and_mask(image,
                        mask = None,
                        **kwargs):
    """Displays an image and its mask side-by-side.

    This method displays an image and its mask side-by-side. Note that if you
    want to simply get the output without displaying it, then you should pass
    `no_show` as True.

    Parameters
    ----------
    image : np.ndarray
        The image to display.
    mask : np.ndarray
        The mask to display.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    A image containing the original image and its mask side-by-side.
    """
    # Get the image and mask from the given input arguments.
    image, mask = resolve_tuple_values(
        image, mask, custom_error =
        "If `image` is a tuple/list, it should contain "
        "two values: the image and its mask.")

    # Prepare the inputs.
    image = format_image(image)
    mask = convert_mask_to_colored_image(format_image(mask, mask = True))

    # Display the image and its mask side-by-side.
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    axes[0].imshow(image)
    axes[1].imshow(mask)
    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout()

    # Display and return the image.
    image = convert_figure_to_image()
    if not kwargs.get('no_show', False):
        _ = display_image(image)
    return image


def show_semantic_segmentation_truth_and_prediction(image,
                                                    real_mask = None,
                                                    predicted_mask = None,
                                                    alpha = 0.3,
                                                    border = True,
                                                    **kwargs):
    # Parse the input arguments to get the corresponding images
    # and masks stored in the correct variables.
    image, real_mask, predicted_mask = resolve_tuple_values(
        image, real_mask, predicted_mask, custom_error =
        "If `image` is a tuple/list, it should contain "
        "three values: the image, the real mask, and the predicted mask.")

    # Generate the real and predicted images with their segmentation masks.
    real_image = annotate_semantic_segmentation(image = image,
                                                mask = real_mask,
                                                alpha = alpha,
                                                border = border)
    predicted_image = annotate_semantic_segmentation(image = image,
                                                     mask = predicted_mask,
                                                     alpha = alpha,
                                                     border = border)

    # Create two side-by-side figures with the images.
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    for ax, img, label in zip(
            axes, (real_image, predicted_image),
            ("Ground Truth Mask", "Predicted Mask")):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(label, fontsize = 15)
        ax.set_aspect('equal')
    fig.tight_layout()

    # Display and return the image.
    image = convert_figure_to_image()
    if not kwargs.get('no_show', False):
        _ = display_image(image)
    return image

