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

import collections

from agml.framework import AgMLSerializable
from agml.utils.data import load_model_benchmarks


# Named tuples which are used by the metadata.
Metric = collections.namedtuple('Metric', ['name', 'value'])


class BenchmarkMetadata(AgMLSerializable):
    """Contains metadata regarding a specific benchmark on a dataset.

    When loading a pretrained model corresponding to a certain benchmark,
    this class is used to provide information regarding the benchmark,
    as well as key hyperparameters which can be used for reproducibility.
    """
    serializable = frozenset(('dataset', 'meta'))

    def __init__(self, dataset):
        # Load the information for the given dataset.
        if dataset is None:
            self._dataset = None
            self._meta = {'hyperparameters': {
                'model_config': None, 'optimizer_config': None,
                'metric': {None: None}, 'epochs_trained': None}}
            return
        try:
            self._dataset = dataset
            self._meta = load_model_benchmarks()[dataset]
        except KeyError:
            raise ValueError(f"Could not find a valid benchmark for dataset ({dataset}).")

    def __str__(self):
        return f"<Benchmark {self._dataset}>({self._meta})"

    @property
    def model_config(self):
        """Returns a dictionary with the configuration for the model."""
        return self._meta['hyperparameters']['model_config']

    @property
    def optimizer_config(self):
        """Returns a dictionary with the configuration for the optimizer."""
        return self._meta['hyperparameters']['optimizer_config']

    @property
    def epochs_trained(self):
        """Returns the number of epochs the benchmark was trained for."""
        return self._meta['hyperparameters']['epochs']

    @property
    def metric(self):
        """Returns the value of the metric for the benchmark."""
        name, value = list(self._meta['metric'].items())[0]
        return Metric(name = name, value = value)



