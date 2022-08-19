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

import io
import re
import sys
import yaml
import collections
from typing import Iterable

from agml.framework import AgMLSerializable
import agml.utils.logging as logging
from agml.utils.general import has_nested_dicts
from agml.utils.data import (
    load_public_sources, load_citation_sources, maybe_you_meant, copyright_print
)


# Named tuples which are used by the metadata.
Tasks = collections.namedtuple('Tasks', ['ml', 'ag'])
Location = collections.namedtuple('Location', ['continent', 'country'])
ImageStats = collections.namedtuple('ImageStats', ['mean', 'std'])


def make_metadata(name, meta = None):
    """Creates the metadata object for the dataset.

    For datasets in the AgML public data repository, this simply returns
    a regular `DatasetMetadata` object with a full set of information for
    the dataset. Otherwise, it returns a `CustomDatasetMetadata` object,
    containing only the provided information in a `meta` dictionary.
    """
    if isinstance(name, (list, tuple, set, frozenset)): # a single dataset
        name = next(iter(name))
    if name in load_public_sources().keys():
        return DatasetMetadata(name)
    return CustomDatasetMetadata(name, meta)


class _MetadataDict(dict):
    """Dictionary subclass that throws a custom error for metadata accesses."""
    def __init__(self, *args, dataset = None, **kwargs):
        if dataset is None:
            raise ValueError(
                "Cannot instantiate metadata dictionary without the dataset name.")
        super(_MetadataDict, self).__init__(*args, **kwargs)
        self._dataset = dataset
        self._custom = kwargs.get('custom', False)

    def __getitem__(self, item):
        try:
            return super(_MetadataDict, self).__getitem__(item)
        except KeyError:
            if self._custom:
                raise KeyError(
                    f"The provided custom dataset '{self._dataset}' has no "
                    f"associated metadata '{item}'. Please provide this when"
                    f"instantiating the loader if you want to use it.")
            raise KeyError(
                f"The dataset '{self._dataset}' is missing metadata '{item}'. "
                f"Please bring this issue to the attention of the AgML team.")


class DatasetMetadata(AgMLSerializable):
    """Stores metadata about a certain AgML dataset.

    When loading in a dataset using the `AgMLDataLoader`, the "info"
    parameter of the class will contain a `DatasetMetadata` object,
    which will expose metadata including (but not limited to):

    - The original dataset source,
    - The location that dataset images were captured,
    - The image and sensor modality, and
    - The data and annotation formats.

    Generally, this can be used as follows:

    > ds = agml.AgMLDataLoader('<dataset-name>')
    > ds.info
    <dataset-name>
    > ds.info.annotation_format
    <annotation-format>

    Most attributes of the dataset will be directly available as a
    property of this class, but any additional info that is not can
    be accessed by treating the `info` object as a dictionary.
    """
    serializable = frozenset(('name', 'metadata', 'citation_meta'))
    is_custom_dataset = False

    def __init__(self, name):
        self._load_source_info(name)

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

    def __fspath__(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, DatasetMetadata):
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other
        return False

    @property
    def data(self):
        return self._metadata

    def _load_source_info(self, name):
        """Loads the data source metadata into the class."""
        if isinstance(name, DatasetMetadata):
            name = name.name

        source_info = load_public_sources()
        if name not in source_info.keys():
            if name.replace('-', '_') not in source_info.keys():
                msg = f"Received invalid public source: '{name}'."
                msg = maybe_you_meant(name, msg)
                raise ValueError(msg)
            else:
                logging.log(
                    f"Interpreted dataset '{name}' as '{name.replace('-', '_')}.'")
                name = name.replace('-', '_')
        self._name = name
        self._metadata = _MetadataDict(
            **source_info[name], dataset = name)

        # Load citation information.
        self._citation_meta = _MetadataDict(
            **load_citation_sources()[name], dataset = name)

    def __getattr__(self, key):
        try:
            # Some weird behavior with lookups can happen.
            return object.__getattribute__(self, key)
        except:
            if key in self._metadata.keys():
                return self._metadata[key]
            raise AttributeError(
                maybe_you_meant(
                    key, f"Received invalid info parameter: '{key}'.",
                    source = self._metadata.keys()))

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def name(self):
        return self._name

    @property
    def num_images(self):
        return int(float(self._metadata['n_images']))

    @property
    def tasks(self):
        """Returns the ML and Agriculture tasks for this dataset."""
        ml_task, ag_task = self._metadata['ml_task'], self._metadata['ag_task']
        return Tasks(ml = ml_task, ag = ag_task)

    @property
    def location(self):
        """Returns the continent and country in which the dataset was made."""
        continent, country = self._metadata['location'].values()
        return Location(continent = continent, country = country)

    @property
    def image_stats(self):
        """Returns the mean and standard deviation of the RGB images."""
        mean, std = self._metadata['stats'].values()
        return ImageStats(mean = mean, std = std)

    @property
    def sensor_modality(self):
        return self._metadata['sensor_modality']

    @property
    def image_format(self):
        return self._metadata['input_data_format']

    @property
    def annotation_format(self):
        return self._metadata['annotation_format']

    @property
    def docs(self):
        return self._metadata['docs_url']

    @property
    def num_to_class(self):
        mapping = self._metadata['classes']
        if has_nested_dicts(mapping):
            out = {}
            for class_type in mapping.keys():
                if isinstance(mapping[class_type], dict):
                    nums = [int(float(i)) for i in mapping[class_type.keys()]]
                    out[class_type] = dict(zip(nums, mapping[class_type].values()))
                else:
                    out[class_type] = mapping[class_type]
            return out
        nums = [int(float(i)) for i in mapping.keys()]
        return dict(zip(nums, mapping.values()))

    @property
    def class_to_num(self):
        mapping = self._metadata['classes']
        if has_nested_dicts(mapping):
            out = {}
            for class_type in mapping.keys():
                if isinstance(mapping[class_type], dict):
                    nums = [int(float(i)) for i in mapping[class_type].keys()]
                    out[class_type] = dict(zip(mapping[class_type].values(), nums))
                else:
                    out[class_type] = mapping[class_type]
            return out
        nums = [int(float(i)) for i in mapping.keys()]
        return dict(zip(mapping.values(), nums))

    @property
    def classes(self):
        classes = self._metadata['classes']
        if has_nested_dicts(classes):
            return {k: list(d.values()) for k, d in classes.items()}
        return list(classes.values())

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def license(self):
        if self._citation_meta['license'] == '':
            return None
        return self._citation_meta['license']

    @property
    def citation(self):
        if self._citation_meta['citation'] == '':
            return None
        return self._citation_meta['citation']

    @property
    def external_image_sources(self):
        return self._metadata['external_image_sources']

    def summary(self):
        """Prints out a summary of the dataset information.

        For all of the properties of information defined in this
        metadata class, this method will print out a summary of all
        of the information in a visually understandable table. Note
        that this does not return anything, so it shouldn't be called
        as `print(loader.info.summary())`, just `loader.info.summary()`.
        """
        def _bold(msg):
            return '\033[1m' + msg + '\033[0m'
        def _bold_yaml(msg): # noqa
            return '<|>' + msg + '<|>'

        _SWITCH_NAMES = {
            'ml_task': "Machine Learning Task",
            'ag_task': "Agricultural Task",
            'real_synthetic': 'Real Or Synthetic',
            'n_images': "Number of Images",
            'docs_url': "Documentation"
        }

        formatted_metadata = {}
        for key, value in self._metadata.items():
            name = key.replace('_', ' ').title()
            if key in _SWITCH_NAMES.keys():
                name = _SWITCH_NAMES[key]
            if name == 'Crop Types':
                value = {int(k): v for k, v in value.items()}
            if name == 'Number of Images':
                value = int(value)
            formatted_metadata[_bold_yaml(name)] = value

        stream = io.StringIO()
        yaml.dump(formatted_metadata, stream, sort_keys = False)
        content = stream.getvalue()
        content = re.sub('<\\|>(.*?)<\\|>', _bold(r'\1'), content)
        header = '=' * 20 + ' DATASET SUMMARY ' + '=' * 20
        print(header)
        print(_bold("Name") + f": {self._name}")
        print(content, end = '')
        print('=' * 57)
        sys.stdout.flush()

    def citation_summary(self):
        """Prints out a summary of the citation information of the dataset.

        This message is the same as is displayed when the dataset is
        initially downloaded, and contains information about the dataset
        license and associated citation (if either exist).
        """
        copyright_print(self._name)


class CustomDatasetMetadata(DatasetMetadata):
    """A metadata wrapper for information given for custom datasets.

    When using a custom dataset in the `AgMLDataLoader`, as there is no
    metadata in the AgML public data source information file, custom
    metadata needs to be passed to the loader instantiation. However,
    this metadata does not necessarily need to be complete to the full
    extent of the rest of the metadata, and so this class is used
    to wrap the provided metadata and infer other arguments instead.
    """
    serializable = frozenset(("name", "metadata"))
    is_custom_dataset = True

    def __init__(self, name, meta): # noqa
        if meta is None:
            msg = f"Received invalid public source: '{name}'."
            out_msg = maybe_you_meant(name, msg)
            if out_msg == msg:
                raise ValueError("Expected metadata when creating a custom loader, got None.")
            else:
                raise ValueError(out_msg)
        self._load_info(name, meta)

    def _load_info(self, name, meta):
        """Loads the provided metadata into the class."""
        # Check that the necessary categories (the task and a list of classes)
        # have been provided, otherwise the loader will not be able to function.
        if 'task' not in meta:
            raise ValueError("Expected a `task` when instantiating a custom loader.")
        if 'classes' not in meta:
            raise ValueError("Expected a list of classes when instantiating a custom loader.")

        # Create a custom dictionary of metadata.
        self._name = name
        self._metadata = _MetadataDict(
            {'ml_task': meta['task'], 'ag_task': meta.get('ag_task', None),
             **meta}, dataset = name)
        self._metadata['classes'] = {str(i + 1): c for i, c in enumerate(meta['classes'])}

        # There is no citation information necessary for custom datasets.
        self._citation_meta = None

    @property
    def license(self):
        raise ValueError("There is no citation information for custom datasets.")

    @property
    def citation(self):
        raise ValueError("There is no citation information for custom datasets.")

    def citation_summary(self): # no citations for custom datasets.
        raise ValueError("There is no citation information for custom datasets.")





