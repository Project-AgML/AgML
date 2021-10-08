"""
A tools module for `agml.viz`, which also serves as almost a
mini-backend to control ops such as the colormap being used.
"""
import os
import json
import functools

import cv2
import numpy as np
from PIL import Image

from agml.backend.tftorch import tf, torch

# Sets the colormaps used in the other `agml.viz` methods.
@functools.lru_cache(maxsize = None)
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

def auto_resolve_image(f):
    """Resolves an image path or image into a read-in image."""
    @functools.wraps(f)
    def _resolver(image, *args, **kwargs):
        if isinstance(image, (str, bytes, os.PathLike)):
            if not os.path.exists(image):
                raise FileNotFoundError(
                    f"The provided image file {image} does not exist.")
            image = cv2.imread(image)
        elif isinstance(image, (list, tuple)):
            if not isinstance(image[0], (str, bytes, os.PathLike)):
                pass
            else:
                processed_images = []
                for image_path in image:
                    if isinstance(image_path, (str, bytes, os.PathLike)):
                        processed_images.append(cv2.imread(image_path))
                    else:
                        processed_images.append(image_path)
                image = processed_images
        return f(image, *args, **kwargs)
    return _resolver

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

