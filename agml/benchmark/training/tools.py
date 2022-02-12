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
from typing import Dict

import pandas as pd

import torch
from torchmetrics import Metric


def gpus(given = None):
    """Gets the number of GPUs to use based on the experiment."""
    if not torch.cuda.is_available():
        return 0
    if given is not None:
        return int(given)
    return torch.cuda.device_count()


def checkpoint_dir(given = None, dataset = None):
    """Returns the directory to save logs/checkpoints to."""
    if given is not None:
        if not os.path.exists(given):
            os.makedirs(given, exist_ok = True)
        return given
    if 'get_ipython()' in globals() and os.path.exists('/content'):
        # In Google Colab.
        return '/content/logs'
    else:
        if os.path.exists('/data2'):
            save_dir = os.path.join(f"/data2/amnjoshi/checkpoints")
        else:
            save_dir = os.path.join(os.path.dirname(__file__), 'logs')
        save_dir = os.path.join(save_dir, dataset)
        os.makedirs(save_dir, exist_ok = True)
        return save_dir


class MetricLogger(object):
    """Logs metrics for training after every epoch."""
    def __init__(self, metrics, file):
        if not isinstance(metrics, dict):
            raise TypeError("Expected a dictionary of metrics.")
        self.metrics: Dict[str, Metric] = metrics

        if not os.path.exists(os.path.dirname(file)):
            raise NotADirectoryError(
                f"The directory of the file {file} does not exist.")
        path_name = os.path.splitext(file)[0]
        file = path_name + ".csv"
        self.out_file = file

        self.log_outputs = []

    def update_metrics(self, *args) -> None:
        raise NotImplementedError()

    def update(self, *args):
        self.update_metrics(*args)

    def compile_epoch(self):
        metric_outs = []
        for metric in self.metrics.values():
            result = metric.compute()
            if isinstance(result, torch.Tensor):
                result = result.item()
            metric_outs.append(result)
        self.log_outputs.append(metric_outs)

    def save(self):
        if not self.log_outputs:
            sys.stderr.write("MetricLogger: No metrics to save!")
            return

        # Compile metrics into a CSV format.
        names = list(self.metrics.keys())
        df = pd.DataFrame(self.log_outputs, columns = names)
        df.to_csv(self.out_file)



