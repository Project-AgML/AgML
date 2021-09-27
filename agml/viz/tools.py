"""
A tools module for `agml.viz`, which also serves as almost a
mini-backend to control ops such as the colormap being used.
"""
import os
import json
import functools

import numpy as np
from PIL import Image

import matplotlib.colors as mcolors

from agml.backend.tftorch import tf, torch

# Sets the colormaps used in the other `agml.viz` methods.
@functools.cache
def _load_colormaps():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           '_assets', 'viz_colormaps.json'), 'r') as f:
        cmaps = json.load(f)
    ret_dict = {}
    for map_ in cmaps.items():
        ret_dict[map_[0]] = {int(k): v for k, v in map_[1].items()}
    return ret_dict

_COLORMAPS = _load_colormaps()
_COLORMAP_CHOICE = 'default'

def get_colormap():
    """Returns the current AgML colormap."""
    global _COLORMAPS, _COLORMAP_CHOICE
    return _COLORMAPS[_COLORMAP_CHOICE]

def format_image(img):
    """Formats an image to be used in a Matplotlib visualization.

    This method is primarily necessary to serve as convenience
    in a few situations: converting images from PyTorch's channels
    first format to channels last, or removing the extra grayscale
    dimension in the case of grayscale images.

    Parameters
    ----------
    img : Any
        An np.ndarray, torch.Tensor, tf.Tensor, or PIL.Image.

    Returns
    -------
    An np.ndarray formatted correctly for a Matplotlib visualization.
    """
    if isinstance(img, Image.Image):
        img = np.array(img.getdata())
    elif isinstance(img, torch.Tensor):
        img = img.numpy()
    elif isinstance(img, np.ndarray):
        img = img
    elif isinstance(img, tf.Tensor):
        img = img.numpy()
    else:
        raise TypeError(
            f"Expected either an np.ndarray, torch.Tensor, "
            f"tf.Tensor, or PIL.Image, got {type(img)}.")

    # Convert channels_first to channels_last.
    if img.ndim == 4:
        if img.shape[0] > 1:
            raise ValueError(
                f"Got a batch of images with shape {img.shape}, "
                f"expected at most a batch of one image.")
        img = np.squeeze(img)
    if img.shape[0] <= 3:
        img = np.transpose(img, (1, 2, 0))

    # Remove the grayscale axis.
    if img.shape[-1] == 1:
        img = np.squeeze(img)

    return img

