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

from rich.console import Console
import numpy as np

from agml.backend.config import data_save_path
from agml.data.metadata import DatasetMetadata
from agml.utils.data import load_public_sources
from agml.utils.downloads import download_dataset as _download  # noqa


class _PublicSourceFilter(object):
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

    _sources = load_public_sources()
    _current_filtered_source = []

    def apply_filters(self, **filters):
        """
        Applies filters to the public data sources.

        Filters the available public datasets based on the provided keyword
        arguments. See the class docstring for details on available filters.

        Parameters
        ----------
        **filters : keyword arguments
            The filters to apply.

        Returns
        -------
        self : _PublicSourceFilter
            Returns the object itself to allow chaining.
        """
        print(self._sources.keys())
        if len(filters) == 0:
            self._current_filtered_source = self._sources.keys()
            return self
        source_sets = []
        for key, value in filters.items():
            if key not in list(self._sources.values())[0].keys():
                raise ValueError(f"Invalid filter: {key}.")
            internal_set = []
            if key == "location":
                self._location_case(value, internal_set)
            elif key in ["n_images", "num_images"] and value.startswith(">"):
                self._n_image_case_greater(int(float(value[1:])), internal_set)
            elif key in ["n_images", "num_images"] and value.startswith("<"):
                self._n_image_case_lesser(int(float(value[1:])), internal_set)
            else:
                for source_ in self._sources.keys():
                    try:
                        if self._sources[source_][key] == value:
                            internal_set.append(source_)
                    except:  # some situations don't have certain arguments.
                        continue
            source_sets.append(internal_set)
        self._current_filtered_source = functools.reduce(np.intersect1d, source_sets)
        return self

    def _location_case(self, desired, value_set):
        """
        Filters datasets based on location.

        This method filters the datasets based on the provided location
        criteria. The `desired` argument should be in the format
        '<location_type>:<location_name>', where `location_type` is either
        'continent' or 'country'.

        Parameters
        ----------
        desired : str
            The desired location criteria.
        value_set : list
            The list to append the matching dataset names to.

        Returns
        -------
        value_set : list
            The updated list of matching dataset names.
        """
        for source_ in self._sources.keys():
            param, value = desired.split(":")
            try:
                if self._sources[source_]["location"][param] == value:
                    value_set.append(source_)
            except KeyError:  # some situations don't have location.
                continue
        return value_set

    def _n_image_case_greater(self, thresh, value_set):
        """
        Filters datasets with a number of images greater than a threshold.

        This method filters the datasets which contain a number of images
        greater than or equal to the provided threshold.

        Parameters
        ----------
        thresh : int
            The threshold for the number of images.
        value_set : list
            The list to append the matching dataset names to.

        Returns
        -------
        value_set : list
            The updated list of matching dataset names.
        """
        for source_ in self._sources.keys():
            try:
                if int(float(self._sources[source_]["n_images"])) >= thresh:
                    value_set.append(source_)
            except KeyError:  # some situations don't have n_images.
                continue
        return

    def _n_image_case_lesser(self, thresh, value_set):
        """
        Filters datasets with a number of images lesser than a threshold.

        This method filters the datasets which contain a number of images
        lesser than or equal to the provided threshold.

        Parameters
        ----------
        thresh : int
            The threshold for the number of images.
        value_set : list
            The list to append the matching dataset names to.

        Returns
        -------
        value_set : list
            The updated list of matching dataset names.
        """
        for source_ in self._sources.keys():
            try:
                if int(float(self._sources[source_]["n_images"])) <= thresh:
                    value_set.append(source_)
            except KeyError:  # some situations don't have n_images.
                continue
        return

    def __repr__(self):
        """Prints a formatted table of the filtered datasets using rich."""

        console = Console()
        if not self._current_filtered_source:
            console.print("[bold yellow]No datasets found matching the criteria.[/]")
            return

        table = Table(title="Filtered Datasets")
        table.add_column("Dataset Name", style="cyan")
        table.add_column("ML Task")
        table.add_column("Location")
        table.add_column("# Images")

        for source_name in self._current_filtered_source:
            meta = self._sources[source_name]
            table.add_row(
                source_name,
                meta.get("ml_task", "N/A"),  # Handle missing 'ml_task'
                f"{meta['location']['continent']}, {meta['location']['country']}" if "location" in meta else "N/A",
                str(meta.get("n_images", "N/A")),  # prints N/A if number of images are not there.
            )

        console.print(table)

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
