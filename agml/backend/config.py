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
import sys
import shutil
import logging

from agml.utils.logging import log

# The base save directory for AgML.
BASE_SAVE_DIR = os.path.join(os.path.expanduser('~'), '.agml')

def default_data_save_path():
    """Returns the default dataset save path for AgML."""
    global BASE_SAVE_DIR
    base_dir = os.path.join(BASE_SAVE_DIR, 'datasets')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def clear_all_datasets():
    """Deletes all of the datasets within the AgML local storage."""
    log("Entering AgML interactive dataset deletion mode.", logging.WARNING)
    msg_format = '\033[91m{0}\033[0m'
    if not input(msg_format.format(
            "Please confirm that you want to delete datasets: [y|n] ")) == "y":
        print("Aborting dataset deletion.")
    local_datasets = os.listdir(default_data_save_path())
    deleted_datasets = []
    for dataset in local_datasets:
        if not os.path.isdir(os.path.join(
                default_data_save_path(), dataset)):
            continue
        if not input(msg_format.format(
                f"Delete dataset '{dataset}'? [y|n] ")) == "y":
            continue
        shutil.rmtree(os.path.join(default_data_save_path(), dataset))
        deleted_datasets.append(dataset)
    print(f"Deleted datasets {deleted_datasets}.")
    log("Exiting AgML interactive dataset deletion mode.", logging.WARNING)

def downloaded_datasets():
    """Lists downloaded datasets in ~/.agml/datasets"""
    return [d for d in os.listdir(
        default_data_save_path()) if os.path.isdir(
        os.path.join(default_data_save_path(), d))]

