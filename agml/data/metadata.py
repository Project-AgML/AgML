import io
import re
import yaml
import collections

import agml.utils.logging as logging
from agml.utils.general import load_public_sources, maybe_you_meant

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

    def __eq__(self, other):
        if isinstance(other, DatasetMetadata):
            return self._name == other._name
        return False

    @property
    def data(self):
        return self._metadata

    def _load_source_info(self, name):
        """Loads the data source metadata into the class."""
        source_info = load_public_sources()
        if name not in source_info.keys():
            if name.replace('-', '_') not in source_info.keys():
                msg = f"Received invalid public source: {name}."
                msg = maybe_you_meant(name, msg)
                raise ValueError(msg)
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


