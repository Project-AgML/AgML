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
Methods for checking the dataset information in AgML.
"""

import os
import pickle

from agml.utils.io import recursive_dirname


_PERSONAL_ACCESS_TOKEN = None


def shape_info_file_contents():
    with open(os.path.join(recursive_dirname(__file__, 2), '_assets', 'shape_info.pickle'), 'rb') as f:
        return pickle.load(f)


def set_git_personal_access_token(token):
    global _PERSONAL_ACCESS_TOKEN
    _PERSONAL_ACCESS_TOKEN = token


def get_personal_access_token():
    global _PERSONAL_ACCESS_TOKEN
    return _PERSONAL_ACCESS_TOKEN
