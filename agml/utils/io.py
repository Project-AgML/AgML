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


# Files which shouldn't be included in a file list.
EXCLUDED_FILES: list = ['.DS_Store']


def _is_valid_file(file):
    """Returns whether a file is valid.

    This means that it is a file and not a directory, but also that
    it isn't an unnecessary dummy file like `.DS_Store` on MacOS.
    """
    global EXCLUDED_FILES
    if os.path.isfile(file):
        if not file.startswith('.git') and file not in EXCLUDED_FILES:
            return True
    return False


def get_file_list(fpath, ext = None, full_paths = True):
    """Gets a list of files from a path."""
    base_list = [os.path.join(fpath, f) if full_paths else f
                 for f in os.listdir(fpath)
                 if _is_valid_file(os.path.join(fpath, f))]
    if ext is not None:
        if isinstance(ext, str):
            return [f for f in base_list if f.endswith(ext)]
        else:
            return [f for f in base_list if
                    any([f.endswith(i) for i in ext])]
    return base_list


def get_dir_list(filepath):
    """Get a list of directories from a path."""
    return [f for f in os.listdir(filepath)
            if os.path.isdir(os.path.join(filepath, f))]


def nested_dir_list(fpath):
    """Returns a nested list of directories from a path."""
    dirs = []
    for f in os.scandir(fpath): # type: os.DirEntry
        if f.is_dir():
            if not os.path.basename(f.path).startswith('.'):
                dirs.append(os.path.join(fpath, f.path))
    if len(dirs) != 0:
        for dir_ in dirs:
            dirs.extend(nested_dir_list(dir_))
    return dirs


def nested_file_list(fpath, ext = None):
    """Returns a nested list of files from a path."""
    files = get_file_list(fpath, ext = ext)
    dirs = nested_dir_list(fpath)
    for dir_ in dirs:
        files.extend(get_file_list(dir_, ext = ext))
    return files


def create_dir(dir_):
    """Creates a directory (or does nothing if it exists)."""
    os.makedirs(dir_, exist_ok = True)


def recursive_dirname(dir_, level = 1):
    """Returns a recursive dirname for the number of levels provided."""
    if level == 0:
        return dir_
    return recursive_dirname(os.path.dirname(dir_), level - 1)


def is_image_file(file):
    """Returns whether a file is an image file."""
    if not isinstance(file, (str, bytes, os.PathLike)):
        return False
    end = os.path.splitext(file)[-1][1:]
    return end.lower() in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']


def load_code_from_string_or_file(code):
    """Returns valid code either from the file `code` or the string `code`."""
    if not isinstance(code, str):
        raise TypeError(
            f"`code` must be either a code string or a path to "
            f"a code file, got {code} of type ({type(code)}).")

    # Check if it is a path. Just because the path doesn't exist doesn't mean
    # that it isn't a path: it could be an incorrect path.
    if os.path.exists(code):
        with open(code, 'r') as f:
            code = f.read()
    else:
        if code.endswith('.cpp') or code.endswith('.cc') \
                or code.endswith('.txt') or code.endswith('.py'):
            raise ValueError(f"The provided file at {code} could not be found.")
    return code

