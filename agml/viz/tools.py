import numpy as np
from PIL import Image

from agml.backend.tftorch import tf, torch

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
    if img.shape[0] <= 3:
        img = np.transpose(img, (1, 2, 0))

    # Remove the grayscale axis.
    if img.shape[-1] == 1:
        img = np.squeeze(img)

    return img

