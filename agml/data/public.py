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

from agml.backend.config import data_save_path
from agml.data.metadata import DatasetMetadata
from agml.utils.data import load_public_sources
from agml.utils.downloads import download_dataset as _download  # noqa


class _PublicSourceFilter:
    """
    Filters public datasets based on provided criteria.

    This class provides a mechanism to filter the available public datasets
    within AgML based on a variety of criteria.  It uses the
    `public_datasources.json` file to access the dataset information and
    performs filtering based on the provided keyword arguments.

    The filtering process supports various criteria, including:

    - `ml_task`: Filters datasets by the machine learning task they are
      designed for (e.g., 'image_classification', 'object_detection').
    - `location`: Filters datasets based on their geographical location.
      The location filter should be specified as 'continent:<continent_name>'
      or 'country:<country_name>'.
    - `n_images` or `num_images`: Filters datasets based on the number of
      images they contain.  Supports exact matching (e.g., `n_images=1000`)
      and threshold-based filtering (e.g., `n_images='>1500'`,
      `num_images='<500'`).  Thresholds are inclusive.

    Examples
    --------
    >>> # Filter for image classification datasets in Africa:
    >>> filtered_sources = _PublicSourceFilter().apply_filters(
    ...     ml_task='image_classification', location='continent:africa'
    ... )
    >>> filtered_datasets = filtered_sources.result()

    >>> # Filter datasets with more than 2000 images:
    >>> many_image_sources = _PublicSourceFilter().apply_filters(n_images='>2000')
    >>> datasets_many_images = many_image_sources.result()

    Methods
    -------
    apply_filters(**filters):
        Applies the specified filters to the list of public datasets.
    print_result():
        Returns a string representation of the filtered dataset names.
    result():
        Returns a list of `DatasetMetadata` objects for the filtered datasets.

    Notes
    -----
    The `public_data_sources` function provides a more user-friendly interface for accessing filtered datasets.
    """
    def __init__(self):
        # Load the public sources (a dict mapping source names to their metadata dict)
        self._sources = load_public_sources()
        self._current_filtered_source = list(self._sources.keys())

    def apply_filters(self, **filters):
        # If no filters are provided, all sources are included.
        if not filters:
            return self

        filtered_sources = []
        for source, meta in self._sources.items():
            if self._matches(meta, filters):
                filtered_sources.append(source)
        self._current_filtered_source = filtered_sources
        return self

    def _matches(self, meta, filters):
        for key, value in filters.items():
            if key == 'location':
                # Expecting a string in the form "param:desired" (e.g., "continent:africa")
                try:
                    loc_key, desired = value.split(':')
                    desired = desired.lower()
                except ValueError:
                    raise ValueError("Location filter must be in the format 'key:value'.")
                if meta.get('location', {}).get(loc_key) != desired:
                    return False
            elif key in ['n_images', 'num_images']:
                try:
                    n_images = int(float(meta.get('n_images', 0)))
                except (ValueError, TypeError):
                    return False
                # Handle threshold conditions like '>1500' or '<1000'
                if isinstance(value, str) and value and value[0] in ('>', '<'):
                    operator = value[0]
                    try:
                        threshold = int(float(value[1:]))
                    except ValueError:
                        raise ValueError("n_images filter threshold must be numeric.")
                    if operator == '>' and n_images < threshold:
                        return False
                    if operator == '<' and n_images > threshold:
                        return False
                else:
                    # Exact match condition
                    if n_images != int(value):
                        return False
            else:
                # For all other filters, perform a simple equality check.
                if meta.get(key) != value:
                    return False
        return True

    def print_result(self):
        return "[%s]" % ", ".join(self._current_filtered_source)

    def result(self):
        """Returns the filtered datasets as DatasetMetadata objects.

        This method returns the final filtered result as a list of
        `DatasetMetadata` objects.  This allows for further inspection
        and access to the metadata of the filtered datasets.

        Returns
        -------
        list
            A list of `DatasetMetadata` objects representing the filtered datasets.
        """
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


def download_public_dataset(dataset, location=None, redownload=False):
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
    location = location if location else data_save_path()
    if not isinstance(dataset, (list, tuple, set, np.ndarray)):
        dataset = [dataset]
    for d in dataset:
        _download(d, location, redownload=redownload)
    return location
