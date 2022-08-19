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

import functools

import numpy as np

from agml.utils.downloads import download_dataset as _download # noqa
from agml.utils.data import load_public_sources
from agml.data.metadata import DatasetMetadata
from agml.backend.config import data_save_path


class _PublicSourceFilter(object):
    """Filters public datasets based on the input filters."""
    _sources = load_public_sources()
    _current_filtered_source = []

    def apply_filters(self, **filters):
        if len(filters) == 0:
            self._current_filtered_source = self._sources.keys()
            return self
        source_sets = []
        for key, value in filters.items():
            if key not in list(self._sources.values())[0].keys():
                raise ValueError(f"Invalid filter: {key}.")
            internal_set = []
            if key == 'location':
                self._location_case(value, internal_set)
            elif key in ['n_images', 'num_images'] and value.startswith('>'):
                self._n_image_case_greater(
                    int(float(value[1:])), internal_set)
            elif key in ['n_images', 'num_images'] and value.startswith('<'):
                self._n_image_case_greater(
                    int(float(value[1:])), internal_set)
            else:
                for source_ in self._sources.keys():
                    try:
                        if self._sources[source_][key] == value:
                            internal_set.append(source_)
                    except: # some situations don't have certain arguments.
                        continue
            source_sets.append(internal_set)
        self._current_filtered_source \
            = functools.reduce(np.intersect1d, source_sets)
        return self

    def _location_case(self, desired, value_set):
        for source_ in self._sources.keys():
            param, value = desired.split(':')
            try:
                if self._sources[source_]['location'][param] == value:
                    value_set.append(source_)
            except KeyError: # some situations don't have location.
                continue
        return value_set

    def _n_image_case_greater(self, thresh, value_set):
        for source_ in self._sources.keys():
            try:
                if int(float(self._sources[source_]['n_images'])) >= thresh:
                    value_set.append(source_)
            except KeyError: # some situations don't have n_images.
                continue
        return

    def _n_image_case_lesser(self, thresh, value_set):
        for source_ in self._sources.keys():
            try:
                if int(float(self._sources[source_]['n_images'])) <= thresh:
                    value_set.append(source_)
            except KeyError: # some situations don't have n_images.
                continue
        return

    def print_result(self):
        return "[%s]" % ', '.join(self._current_filtered_source)

    def result(self):
        return [DatasetMetadata(s) for s in self._current_filtered_source]


def public_data_sources(**filters):
    """Lists all of the public data sources in AgML.

    The `filters` argument can be used to get a list of datasets
    which are constrained to a specific value, such as in the following:

    Get a list of all image classification datasets in AgML.
    > public_data_sources(ml_task = 'image_classification')

    The `location` and `n_images` arguments have a different procedure.
    For location, the argument needs to be passed as follows:

    > public_data_sources(location = 'continent:africa')
    > public_data_sources(location = 'country:denmark')

    For number of images, the argument can either be an exact integer,
    or if checking a threshold, then it should be passed as follows
    (note that the threshold is inclusive):

    > public_data_sources(n_images = 1000)
    > public_data_sources(n_images = ">1500")

    This method returns a list of `DatasetMetadata` objects, which can
    then be further inspected by additional conditions.

    Parameters
    ----------
    filters : Any
        An arbitrary number of keyword arguments representing different
        filters which should be applied to the list of data sources.

    Returns
    -------
    A list of sources filtered by the passed arguments.
    """
    s = _PublicSourceFilter().apply_filters(**filters)
    return s.result()


def source(name):
    """Returns the metadata of a single public dataset source.

    If you just want to inspect the info of one dataset, then you can use
    this method with `name` representing a dataset to get a single piece
    of `DatasetMetadata` which contains its information.

    Parameters
    ----------
    name : str
        The name of the dataset you want to inspect.

    Returns
    -------
    The metadata of the dataset `name`.
    """
    return DatasetMetadata(name)


def download_public_dataset(dataset, location = None, redownload = False):
    """Downloads a public dataset from AgML to a directory.

    While the `AgMLDataLoader` exists to load data directly into an
    accessor class, if you want to simply download the data to
    a location, whether for inspection or other purposes, this
    method enables such downloading. By default, data will be
    downloaded to the `~/.agml/datasets` directory.

    If you want to download multiple datasets at once, simply pass
    a sequence of datasets to the `dataset` argument.

    Parameters
    ----------
    dataset : str, list
        The name of the dataset(s) you want to download.
    location : str
        The local location you want to download the data to.
    redownload : bool
        Whether to download the dataset irrespective of existence.

    Returns
    -------
    The local directory of the dataset.
    """
    location = location if location \
                        else data_save_path()
    if not isinstance(dataset, (list, tuple, set, np.ndarray)):
        dataset = [dataset]
    for d in dataset:
        _download(d, location, redownload = redownload)
    return location


