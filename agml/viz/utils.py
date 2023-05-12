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
import functools

import cv2
import matplotlib.pyplot as plt

from agml.utils.image import imread_context


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

