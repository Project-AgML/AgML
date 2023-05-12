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

from agml.viz.tools import get_viz_backend


def display_image(image, **kwargs):
    """Displays an image using the appropriate backend."""
    if get_viz_backend() == 'cv2':
        # If running in Colab, then use a separate procedure.
        if 'google.colab' in sys.modules:
            try:
                from google.colab.patches import cv2_imshow
            except ImportError:
                raise ImportError(
                    "Can't import cv2_imshow from google.colab.patches. "
                    "Use the matplotlib backend for `agml.viz`.")
            cv2_imshow(image)
            return

        if kwargs.get('read_raw', False):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert back to BGR
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyWindow('image')

    if get_viz_backend() == 'matplotlib':
        plt.figure(figsize = (10, 10))
        plt.imshow(image)
        plt.gca().axis('off')
        plt.gca().set_aspect('equal')
        plt.show()


