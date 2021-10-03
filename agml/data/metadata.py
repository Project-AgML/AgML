import json
import functools
import collections

import agml.utils.logging as logging
from agml.utils.general import load_public_sources

class DatasetMetadata(object):
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
    def __init__(self, name):
        self._load_source_info(name)

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

    @property
    def data(self):
        return self._metadata

    @functools.lru_cache(maxsize = None)
    def _load_source_info(self, name):
        """Loads the data source metadata into the class."""
        source_info = load_public_sources()
        if name not in source_info.keys():
            if name.replace('-', '_') not in source_info.keys():
                raise ValueError(f"Received invalid public source: {name}.")
            else:
                logging.log(
                    f"Interpreted dataset '{name}' as '{name.replace('-', '_')}.'")
                name = name.replace('-', '_')
        self._name = name
        self._metadata = source_info[name]

    def __getattr__(self, key):
        if key in self._metadata.keys():
            return self._metadata[key]
        raise AttributeError(f"Received invalid info parameter: {key}.")

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
        Tasks = collections.namedtuple('Tasks', ['ml', 'ag'])
        ml_task, ag_task = self._metadata['ml_task'], self._metadata['ag_task']
        return Tasks(ml = ml_task, ag = ag_task)

    @property
    def location(self):
        """Returns the continent and country in which the dataset was made."""
        Location = collections.namedtuple('Location', ['continent', 'country'])
        continent, country = self._metadata['location'].values()
        return Location(continent = continent, country = country)

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
        mapping = self._metadata['crop_types']
        nums = [int(float(i)) for i in mapping.keys()]
        return dict(zip(nums, mapping.values()))

    @property
    def class_to_num(self):
        mapping = self._metadata['crop_types']
        nums = [int(float(i)) for i in mapping.keys()]
        return dict(zip(mapping.values(), nums))


