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
import shutil

import pytest

BASE_PATH = os.path.join(os.path.expanduser('~'), '.agml', 'datasets')
_USED_DATASETS = ['bean_disease_uganda', 'apple_flower_segmentation', 'apple_detection_usa']
_PREEXISTING_DATASETS = []


def _find_preexisting_datasets():
    global _USED_DATASETS, _PREEXISTING_DATASETS, BASE_PATH
    for dataset in _USED_DATASETS:
        if os.path.exists(os.path.join(BASE_PATH, dataset)) \
                and os.path.isdir(os.path.join(BASE_PATH, dataset)):
            _PREEXISTING_DATASETS.append(dataset)


def _remove_new_datasets():
    global _USED_DATASETS, _PREEXISTING_DATASETS, BASE_PATH
    for dataset in _USED_DATASETS:
        if dataset not in _PREEXISTING_DATASETS:
            shutil.rmtree(dataset)


def pre_test_configure():
    _find_preexisting_datasets()


def post_test_cleanup():
    _remove_new_datasets()


def execute_tests():
    pre_test_configure()
    pytest.main()
    post_test_cleanup()


if __name__ == '__main__':
    execute_tests()
