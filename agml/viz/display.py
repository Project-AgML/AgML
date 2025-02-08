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

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ExifTags, Image

from agml.viz.tools import get_viz_backend


def correct_image_orientation(image):
    """Correct image orientation based on EXIF data."""
    try:
        pil_image = Image.fromarray(image)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = pil_image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)

            # Rotate the image according to EXIF orientation
            if orientation_value == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation_value == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation_value == 8:
                pil_image = pil_image.rotate(90, expand=True)

        # Convert back to numpy array for further use (e.g., with OpenCV or Matplotlib)
        return np.array(pil_image)
    except Exception as e:
        # If any issue occurs, return the image as-is
        return image


def display_image(image, **kwargs):
    """Displays an image using the appropriate backend."""
    notebook = False
    try:
        shell = eval("get_ipython().__class__.__name__")
        if shell == "ZMQInteractiveShell":
            notebook = True
    except NameError:
        pass

    if get_viz_backend() == "cv2":
        # If running in Colab, then use a separate procedure.
        if "google.colab" in sys.modules:
            try:
                from google.colab.patches import cv2_imshow
            except ImportError:
                raise ImportError(
                    "Can't import cv2_imshow from google.colab.patches. " "Use the matplotlib backend for `agml.viz`."
                )
            cv2_imshow(image)
            return

        # If running in a Jupyter notebook, then for some weird reason it automatically
        # displays images in the background, so don't actually do anything here.
        if notebook:
            # If the input content is not a figure, then we can display it.
            if kwargs.get("matplotlib_figure", True):
                return

        else:
            if kwargs.get("read_raw", False):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert back to BGR
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyWindow("image")
            return

    if get_viz_backend() == "matplotlib":
        # If we are in a notebook or Colab, then don't show anything (since
        # it will be displayed automatically due to the way notebooks are).
        #
        # But also, some methods don't have any Matplotlib functions - so
        # for those we bypass this skip just to show the result.
        if "google.colab" in sys.modules or notebook and not kwargs.get("force_show", False):
            return

    # Default case is matplotlib, since it is the most modular.
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.show()
