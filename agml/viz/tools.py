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

"""
A tools module for `agml.viz`, which also serves as almost a
mini-backend to control ops such as the colormap being used.
"""
import os
import io
import json
import functools

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from agml.backend.tftorch import tf, torch
from agml.utils.logging import log
from agml.utils.image import imread_context


# Sets the colormaps used in the other `agml.viz` methods.
@functools.lru_cache(maxsize = None)
def _load_colormaps():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           '_assets', 'viz_colormaps.json'), 'r') as f:
        cmaps = json.load(f)
    ret_dict = {}
    for map_ in cmaps.items():
        ret_dict[map_[0]] = map_[1] * 5
    return ret_dict


_COLORMAPS = _load_colormaps()
_COLORMAP_CHOICE = 'default'


def get_colormap():
    """Returns the current AgML colormap."""
    global _COLORMAPS, _COLORMAP_CHOICE
    return _COLORMAPS[_COLORMAP_CHOICE]


def set_colormap(colormap):
    """Sets the current AgML colormap used in color displays.

    This method accepts one argument, `colormap`, which can be
    any of the colormaps listed in `_assets/viz_colormaps.json`,
    namely one of the following:

    1. "default": Traditional matplotlib RGB colors.
    2. "agriculture": Various shades of green (for agriculture).

    Parameters
    ----------
    colormap : str
        The colormap to set.
    """
    global _COLORMAP_CHOICE, _COLORMAPS
    colormap = colormap.lower()
    if colormap not in _COLORMAPS.keys():
        raise ValueError(f"Invalid colormap {colormap} received.")
    _COLORMAP_CHOICE = colormap


def auto_resolve_image(f):
    """Resolves an image path or image into a read-in image."""
    @functools.wraps(f)
    def _resolver(image, *args, **kwargs):
        if isinstance(image, (str, bytes, os.PathLike)):
            if not os.path.exists(image):
                raise FileNotFoundError(
                    f"The provided image file {image} does not exist.")
            with imread_context(image) as img:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, (list, tuple)):
            if not isinstance(image[0], (str, bytes, os.PathLike)):
                pass
            else:
                processed_images = []
                for image_path in image:
                    if isinstance(image_path, (str, bytes, os.PathLike)):
                        with imread_context(image_path) as img:
                            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        processed_images.append(image)
                    else:
                        processed_images.append(image_path)
                image = processed_images
        return f(image, *args, **kwargs)
    return _resolver


def show_when_allowed(f):
    """Stops running `plt.show()` when in a Jupyter Notebook."""
    _in_notebook = False
    try:
        shell = eval("get_ipython().__class__.__name__")
        cls = eval("get_ipython().__class__")
        if shell == 'ZMQInteractiveShell' or 'colab' in cls:
            _in_notebook = True
    except:
        pass

    @functools.wraps(f)
    def _cancel_display(*args, **kwargs):
        show = kwargs.pop('show', True)
        res = f(*args, **kwargs)
        if not _in_notebook and show:
            plt.show()
        return res
    return _cancel_display


def format_image(img, mask = False):
    """Formats an image to be used in a Matplotlib visualization.

    This method takes in one of a number of common image/array types
    and returns a formatted NumPy array with formatted image data
    as expected by matplotlib.

    This method is primarily necessary to serve as convenience
    in a few situations: converting images from PyTorch's channels
    first format to channels last, or removing the extra grayscale
    dimension in the case of grayscale images.

    Parameters
    ----------
    img : Any
        An np.ndarray, torch.Tensor, tf.Tensor, or PIL.Image.
    mask : Any
        Whether the image is a segmentation mask.

    Returns
    -------
    An np.ndarray formatted correctly for a Matplotlib visualization.
    """
    # Get the numpy array from the image type.
    if isinstance(img, np.ndarray):
        img = img
    elif isinstance(img, Image.Image):
        img = np.array(img).reshape((img.height, img.width, len(img.getbands())))
    elif isinstance(img, torch.Tensor):
        if img.is_cuda:
            img = img.cpu().detach().numpy()
        img = img.numpy()
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

    # If the image is in range 0-255 but a float image, then
    # we need to convert it to an integer type.
    if not mask:
        if np.issubdtype(img.dtype, np.inexact):
            if not img.max() <= 1: # noqa
                img = img.astype(np.uint8)
            else:
                img = (img * 255).astype(np.uint8)

        # Convert 64-bit integer to unsigned 8-bit.
        if img.dtype == np.int64:
            log("Converting image of dtype `np.int64` to `np.uint8` for display. "
                "This may cause a loss in precision/invalid result.")
            img = img.astype(np.uint8)

    # Return the formatted image.
    return img


def convert_figure_to_image(fig = None):
    """This method is used to convert a Matplotlib figure to an image array."""
    # Use PIL to get the image, then convert to an array.
    buf = io.BytesIO()
    fig = fig if fig is not None else plt.gcf()
    fig.savefig(buf, format = 'png')
    buf.seek(0)
    arr = np.fromstring(buf.read(), dtype = np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

