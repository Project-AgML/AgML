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

import random
import inspect

import cv2

from agml.utils.io import (
    get_file_list as _get_file_list,
    get_dir_list as _get_dir_list,
    nested_file_list as _get_nested_file_list,
    nested_dir_list as _get_nested_dir_list,
    recursive_dirname as _recursive_dirname
)


def get_file_list(path, ext = None, nested = False):
    """Returns a list of files in a directory.

    This function returns a list of files in a directory, optionally
    filtered by extension. The `path` argument is the path to the
    directory, and the `ext` argument is the file extension to filter
    by. If `ext` is `None`, then all files in the directory are
    returned.

    Args:
        path (str): The path to the directory.
        ext (str, optional): The file extension to filter by.
        nested (bool, optional): Whether to recursively search the directory.

    Returns:
        list: A list of files in the directory, optionally filtered by
            extension.
    """
    if nested:
        return _get_nested_file_list(path, ext = ext)
    return _get_file_list(path, ext = ext)


def get_dir_list(path, nested = False):
    """Returns a list of directories in a directory.

    This function returns a list of directories in a directory,
    optionally recursively. The `path` argument is the path to the
    directory.

    Args:
        path (str): The path to the directory.
        nested (bool, optional): Whether to recursively search the directory.

    Returns:
        list: A list of directories in the directory.
    """
    if nested:
        return _get_nested_dir_list(path)
    return _get_dir_list(path)


def recursive_dirname(path, depth = 1):
    """Returns the directory name of a path.

    This function returns the directory name of a path, optionally
    recursively. The `path` argument is the path to the directory, and
    the `depth` argument is the number of times to recursively get the
    directory name.

    Args:
        path (str): The path to the directory.
        depth (int, optional): The number of times to recursively get
            the directory name.

    Returns:
        str: The directory name of the path.
    """
    return _recursive_dirname(path, level = depth)


def parent_path(depth):
    """Returns the parent folder at level `depth` of the current file.

    Args:
        depth (int): The number of levels to go up.

    Returns:
        str: The path to the parent folder.
    """
    frame = inspect.stack()[1]
    fname = inspect.getmodule(frame[0]).__file__
    return _recursive_dirname(fname, level = depth)


def random_file(path, **kwargs):
    """Returns a random file from a directory.

    Args:
        path (str): The path to the directory.
        **kwargs: Keyword arguments to pass to `get_file_list`.

    Returns:
        str: The path to the random file.
    """
    return random.choice(get_file_list(path, **kwargs))


def read_image(path, **kwargs):
    """Reads an image from a file.

    Args:
        path (str): The path to the image file.
        **kwargs: Keyword arguments to pass to `cv2.imread`.

    Returns:
        numpy.ndarray: The image.
    """
    return cv2.imread(path, **kwargs)

