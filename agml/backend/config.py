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
import json
import shutil
import logging

from agml.utils.logging import log


# The super base save directory for AgML. This is the overriding base
# save directory and default original save directory. This is saved here
# because the default path to save datasets to can be overridden by
# `set_dataset_save_path()`, but we need to keep the super directory to
# access the config file which stores this information in the first place.
SUPER_BASE_DIR = os.path.join(os.path.expanduser('~'), '.agml')


# This is the path to the saved datasets. By default, this saves to
# SUPER_BASE_DIR/datasets, but can be overridden. The value of this
# is set upon the instantiation of the module (see below code).
DATASET_SAVE_DIR: str



# This is similar to `DATASET_SAVE_DIR`, but is for synthetically generated
# datasets using Helios. By default, this will be SUPER_BASE_DIR/synthetic,
# but it can be overridden. The value is set upon instantiation of the module.
SYNTHETIC_SAVE_DIR: str


# This is the path to any downloaded models. By default, this saves to
# SUPER_BASE_DIR/models, but can be overridden. The value is set upon
# instantiation of the module (see the below code).
MODEL_SAVE_DIR: str


# Loads the configuration info. We don't cache this since it may
# change if the user decides to change the path. This method
# also runs upon the first import of AgML to set it properly.
def _load_config_info():
    global DATASET_SAVE_DIR, SYNTHETIC_SAVE_DIR, MODEL_SAVE_DIR
    try:
        with open(os.path.join(SUPER_BASE_DIR, 'config.json'), 'r') as f:
            contents = json.load(f)
            DATASET_SAVE_DIR = contents['data_path']
            SYNTHETIC_SAVE_DIR = contents['synthetic_data_path']
            MODEL_SAVE_DIR = contents['model_path']
    except (OSError, KeyError):
        with open(os.path.join(SUPER_BASE_DIR, 'config.json'), 'w') as f:
            json.dump({'data_path': os.path.join(SUPER_BASE_DIR, 'datasets'),
                       'synthetic_data_path': os.path.join(
                           SUPER_BASE_DIR, 'synthetic'),
                       'model_path': os.path.join(SUPER_BASE_DIR, 'models')}, f)
        _load_config_info()
_load_config_info()


def data_save_path():
    """Returns the default dataset save path for AgML."""
    global DATASET_SAVE_DIR
    return DATASET_SAVE_DIR


def set_data_save_path(location = None):
    """Sets the default dataset save path for AgML.
    Changing the data save path using this method permanently changes
    the data save path for all future sessions, until it is changed
    or switched back to the original. If you just want to download one
    dataset to a different path, use the `dataset_path` argument.
    Parameters
    ----------
    location : str
        The location to save the data to.
    Returns
    -------
    The fully expanded location.
    """
    global SUPER_BASE_DIR
    if location is None or location == 'reset':
        location = os.path.join(SUPER_BASE_DIR, 'datasets')
    location = os.path.expanduser(location)
    if not os.path.exists(location) and not os.path.isdir(location):
        raise NotADirectoryError(
            f"The provided destination {location} does "
            f"not exist, or is not a directory.")
    with open(os.path.join(SUPER_BASE_DIR, 'config.json'), 'r') as f:
        contents = json.load(f)
    contents['data_path'] = os.path.realpath(os.path.abspath(location))
    with open(os.path.join(SUPER_BASE_DIR, 'config.json'), 'w') as f:
        json.dump(contents, f)
    return


def synthetic_data_save_path():
    """Returns the default synthetic data save path for AgML."""
    global SYNTHETIC_SAVE_DIR
    return SYNTHETIC_SAVE_DIR


def set_synthetic_save_path(location = None):
    """Sets the default synthetic data save path for AgML.
    Changing the data save path using this method permanently changes
    the data save path for all future sessions, until it is changed
    or switched back to the original. If you just want to download one
    dataset to a different path, use the `dataset_path` argument.
    Parameters
    ----------
    location : str
        The location to save the data to.
    Returns
    -------
    The fully expanded location.
    """
    global SUPER_BASE_DIR
    if location is None or location == 'reset':
        location = os.path.join(SUPER_BASE_DIR, 'synthetic')
    location = os.path.expanduser(location)
    if not os.path.exists(location) and not os.path.isdir(location):
        raise NotADirectoryError(
            f"The provided destination {location} does "
            f"not exist, or is not a directory.")
    with open(os.path.join(SUPER_BASE_DIR, 'config.json'), 'r') as f:
        contents = json.load(f)
    contents['synthetic_data_path'] = os.path.realpath(os.path.abspath(location))
    with open(os.path.join(SUPER_BASE_DIR, 'config.json'), 'w') as f:
        json.dump(contents, f)
    return


def model_save_path():
    """Returns the default model save path for AgML."""
    global MODEL_SAVE_DIR
    return MODEL_SAVE_DIR


def set_model_save_path(location = None):
    """Sets the default model save path for AgML.
    Changing the data save path using this method permanently changes
    the data save path for all future sessions, until it is changed
    or switched back to the original.
    Parameters
    ----------
    location : str
        The location to save the model to.
    Returns
    -------
    The fully expanded location.
    """
    global SUPER_BASE_DIR
    if location is None or location == 'reset':
        location = os.path.join(SUPER_BASE_DIR, 'models')
    location = os.path.expanduser(location)
    if not os.path.exists(location) and not os.path.isdir(location):
        raise NotADirectoryError(
            f"The provided destination {location} does "
            f"not exist, or is not a directory.")
    with open(os.path.join(SUPER_BASE_DIR, 'config.json'), 'r') as f:
        contents = json.load(f)
    contents['model_path'] = os.path.realpath(os.path.abspath(location))
    with open(os.path.join(SUPER_BASE_DIR, 'config.json'), 'w') as f:
        json.dump(contents, f)
    return



def clear_all_datasets():
    """Deletes all of the datasets within the AgML local storage."""
    log("Entering AgML interactive dataset deletion mode.", logging.WARNING)
    msg_format = '\033[91m{0}\033[0m'
    if not input(msg_format.format(
            "Please confirm that you want to delete datasets: [y|n] ")) == "y":
        print("Aborting dataset deletion.")
    local_datasets = os.listdir(data_save_path())
    deleted_datasets = []
    for dataset in local_datasets:
        if not os.path.isdir(os.path.join(
                data_save_path(), dataset)):
            continue
        if not input(msg_format.format(
                f"Delete dataset '{dataset}'? [y|n] ")) == "y":
            continue
        shutil.rmtree(os.path.join(data_save_path(), dataset))
        deleted_datasets.append(dataset)
    print(f"Deleted datasets {deleted_datasets}.")
    log("Exiting AgML interactive dataset deletion mode.", logging.WARNING)


def downloaded_datasets():
    """Lists downloaded datasets in ~/.agml/datasets"""
    return [d for d in os.listdir(
        data_save_path()) if os.path.isdir(
        os.path.join(data_save_path(), d))]