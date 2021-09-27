import os

import numpy as np

import matplotlib.pyplot as plt

from agml.viz.tools import format_image, get_colormap

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
    if isinstance(image, (list, tuple)) and not mask:
        try:
            image, mask = image
        except ValueError:
            raise ValueError(
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



