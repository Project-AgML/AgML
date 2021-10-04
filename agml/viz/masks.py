import os

import cv2
import numpy as np

import matplotlib.pyplot as plt

from agml.utils.general import resolve_tuple_values
from agml.viz.tools import format_image, get_colormap, auto_resolve_image

def _preprocess_mask(mask):
    """Preprocesses a mask with a distinct colorscheme."""
    if mask.ndim == 2:
        mask = np.dstack((mask, mask, mask))

    # Add a colorscheme to the labels.
    label_indx = 1
    while True:
        x_values, y_values, *_ = np.where(mask == label_indx)
        coords = np.stack([x_values, y_values]).T
        if coords.size == 0:
            break
        for c in coords:
            mask[c[0]][c[1]] = get_colormap()[label_indx]
        label_indx += 1
    return mask

def _mask_2d_to_3d(mask):
    """Converts a mask from 2-dimensional to channel-by-channel."""
    channels = np.unique(mask)[1:]
    out = np.zeros(shape = (*mask.shape[:2], max(channels)))
    iter_idx = 1
    while True:
        if iter_idx > max(channels):
            break
        elif iter_idx not in channels:
            out[:, :, iter_idx - 1] = np.zeros(shape = (mask.shape[:2]))
            iter_idx += 1
            continue
        coords = np.stack(np.where(mask == iter_idx)[:2]).T
        channel = np.zeros(shape = (*mask.shape[:2],))
        channel[tuple([*coords.T])] = 1
        out[:, :, iter_idx - 1] = channel
        iter_idx += 1
    return out

def output_to_mask(mask):
    """Converts an output annotation mask into a visual segmentation mask.

    Given the output segmentation mask from a model (a 2-dimensional
    image array with a number representing a class label) this method
    will convert it into an image with colors for the regions with
    different labels. This is a purely visual transformation.

    Parameters
    ----------
    mask : np.ndarray
        The above cases of a possible segmentation mask.

    Returns
    -------
    The colorful visually formatted mask.
    """
    return _preprocess_mask(mask)


@auto_resolve_image
def visualize_image_and_mask(image, mask = None):
    """Visualizes an image and its corresponding segmentation mask.

    Creates a 1 row, 2 column frame and displays an image with
    its corresponding segmentation mask. Applies a distinct
    colorscheme to the mask to visualize colors more clearly.

    Parameters
    ----------
    image : Any
        Either the original image, or a tuple containing the image
        and its mask (if using a DataLoader, for example).
    mask : np.ndarray
        The output mask.

    Returns
    -------
    The matplotlib figure with the images.
    """
    image, mask = resolve_tuple_values(
        image, mask, custom_error =
        "If `image` is a tuple/list, it should contain "
        "two values: the image and its mask.")
    image = format_image(image)
    mask = _preprocess_mask(format_image(mask))

    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    axes[0].imshow(image)
    axes[1].imshow(mask)
    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
    plt.show()
    return fig


@auto_resolve_image
def overlay_segmentation_masks(image, mask = None, border = True):
    """Overlays segmentation masks over an image.

    Creates a single image and displays the segmentation annotations
    from the mask overlaid onto the main image. The mask will be
    semi-transparent and have a colorscheme applied to it in order
    to facilitate easier visualization of it.

    Parameters
    ----------
    image : Any
        Either the original image, or a tuple containing the image
        and its mask (if using a DataLoader, for example).
    mask : np.ndarray
        The output mask (should be an array with dimension 2).
    border : bool
        Whether to add a border to the segmentation overlay.

    Returns
    -------
    A `np.ndarray` representing the image.
    """
    image, mask = resolve_tuple_values(
        image, mask, custom_error =
        "If `image` is a tuple/list, it should contain "
        "two values: the image and its mask.")
    image = format_image(image)
    mask = _mask_2d_to_3d(mask)

    # Plot the segmentation masks over the image
    mask = np.transpose(mask, (2, 0, 1))
    for level, label in enumerate(mask):
        label = np.expand_dims(label, axis = -1)
        label = (label * (255 / (np.max(label, axis = (0, 1))))) \
                .astype(np.uint8)

        # Find contours and plot them
        _, thresh = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for indx, contour in enumerate(contours):
            color = get_colormap()[level + 1]
            overlay = image.copy()
            cv2.fillPoly(overlay, pts = [contour], color = color)
            label = label.astype(np.float32)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            if border:
                image = cv2.polylines(
                    image, pts = [contour], isClosed = True,
                    color = color, thickness = 2)
    return image


@auto_resolve_image
def visualize_overlaid_masks(image, mask = None):
    """Displays an image with segmentation masks overlaid on it.

    See `overlay_segmentation_masks` for an explanation of procedure.

    Parameters
    ----------
    image : Any
        Either the original image, or a tuple containing the image
        and its mask (if using a DataLoader, for example).
    mask : np.ndarray
        The output mask (should be an array with dimension 2).

    Returns
    -------
    A `np.ndarray` representing the image.
    """
    # Plot the segmentation masks over the image
    image = overlay_segmentation_masks(image, mask)

    # Plot the figure
    plt.figure(figsize = (10, 10))
    plt.imshow(image)
    plt.gca().axis('off')
    plt.gca().set_aspect('equal')
    plt.show()
    return image





