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

import cv2


def consistent_shapes(objects):
    """Determines whether the shapes of the objects are consistent."""
    try:
        shapes = [i.shape for i in objects]
        return shapes.count(shapes[0]) == len(shapes)
    except:
        try:
            lens = [len(i) for i in objects]
            return lens.count(lens[0]) == len(lens)
        except:
            if isinstance(objects[0], (int, float)):
                return True
            return False


def needs_batch_dim(image):
    """Determines whether an image has or is missing a batch dimension."""
    if not hasattr(image, 'shape'):
        raise TypeError(
            "Can only determine batch dimensions for numpy arrays or tensors.")
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3:
        if image.shape[0] != 1:
            return True
    return False


def resolve_image_size(size):
    if isinstance(size, int):
        return size, size
    elif isinstance(size, (list, tuple, set, np.ndarray)):
        if not len(size) == 2:
            raise ValueError(
                "Only two values must be provided for an input image size.")
        return size
    else:
        raise ValueError(
            f"Expected either an integer or list of two values for "
            f"image size, got ({size}) of type ({type(size)}).")


class imread_context(object):
    """Wraps the `cv2.imread` function into a context.

    This context allows for a better error message regarding the
    loading of images using OpenCV. Traditionally, if the path
    doesn't exist or the image is empty, the `imread` function
    will just return `None`, which will usually lead to a cryptic
    error message in future operations. This context catches such
    issues and raises a more detailed error message for the user.
    """
    def __init__(self, path, flags = None):
        self._path = path
        if not os.path.exists(path):
            raise ValueError(
                f"The image path '{path}' does not exist. "
                f"Please check the dataset you are using, "
                f"and if files are missing, re-download it.")
        self.flags = flags
        if self.flags is None:
            self.flags = cv2.IMREAD_UNCHANGED

    def __enter__(self):
        try:

            img = cv2.imread(self._path, self.flags)
        except cv2.error:
            raise ValueError(
                f"An error was encountered when loading the image "
                f"'{self._path}'. Please check the dataset you are "
                f"using and re-download it if files are missing.")
        else:
            if img is None:
                raise ValueError(
                    f"The image at '{self._path}' is empty,"
                    f" corrupted, or could not be read. Please "
                    f"re-download the dataset you are using.")
        return img

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


