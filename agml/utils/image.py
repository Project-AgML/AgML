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

class imread_context(object):
    """Wraps the `cv2.imread` function into a context.

    This context allows for a better error message regarding the
    loading of images using OpenCV. Traditionally, if the path
    doesn't exist or the image is empty, the `imread` function
    will just return `None`, which will usually lead to a cryptic
    error message in future operations. This context catches such
    issues and raises a more detailed error message for the user.
    """
    def __init__(self, path):
        self._path = path
        if not os.path.exists(path):
            raise ValueError(
                f"The image path '{path}' does not exist. "
                f"Please check the dataset you are using, "
                f"and if files are missing, re-download it.")

    def __enter__(self):
        try:
            img = cv2.imread(self._path, cv2.IMREAD_UNCHANGED)
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

