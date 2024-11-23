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
from functools import wraps
from typing import Callable, Dict

import pandas as pd
import torch
from torchmetrics import Metric


def gpus(given=None):
    """Gets the number of GPUs to use based on the experiment."""
    if not torch.cuda.is_available():
        return 0
    if given is not None:
        return int(given)
    return torch.cuda.device_count()


def checkpoint_dir(given=None, dataset=None):
    """Returns the directory to save logs/checkpoints to."""
    if given is not None:
        if given.endswith("dataset"):
            # if a path is passed like /root/dataset, this updates to the name
            given = os.path.dirname(given)
            given = os.path.join(given, dataset)
        if not os.path.exists(given):
            os.makedirs(given, exist_ok=True)
        return given
    if "get_ipython()" in globals() and os.path.exists("/content"):
        # In Google Colab.
        return "/content/logs"
    else:
        if os.path.exists("/data2"):
            save_dir = os.path.join(f"/data2/amnjoshi/checkpoints")
        else:
            save_dir = os.path.join(os.path.dirname(__file__), "logs")
        save_dir = os.path.join(save_dir, dataset)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir


class MetricLogger(object):
    """Logs metrics for training after every epoch."""

    def __init__(self, metrics, file):
        if not isinstance(metrics, dict):
            raise TypeError("Expected a dictionary of metrics.")
        self.metrics: Dict[str, Metric] = metrics

        if not os.path.exists(os.path.dirname(file)):
            raise NotADirectoryError(
                f"The directory of the file {file} does not exist."
            )
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
        df = pd.DataFrame(self.log_outputs, columns=names)
        df.to_csv(self.out_file)


# Ported from PyTorch Lightning v1.3.0.
def auto_move_data(fn: Callable) -> Callable:
    """
    Decorator for :class:`~pytorch_lightning.core.lightning.LightningModule` methods for which
    input arguments should be moved automatically to the correct device.
    It as no effect if applied to a method of an object that is not an instance of
    :class:`~pytorch_lightning.core.lightning.LightningModule` and is typically applied to ``__call__``
    or ``forward``.

    Args:
        fn: A LightningModule method for which the arguments should be moved to the device
            the parameters are on.

    Example::

        # directly in the source code
        class LitModel(LightningModule):

            @auto_move_data
            def forward(self, x):
                return x

        # or outside
        LitModel.forward = auto_move_data(LitModel.forward)

        model = LitModel()
        model = model.to('cuda')
        model(torch.zeros(1, 3))

        # input gets moved to device
        # tensor([[0., 0., 0.]], device='cuda:0')

    """

    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        from pytorch_lightning import LightningModule

        if not isinstance(self, LightningModule):
            return fn(self, *args, **kwargs)

        args, kwargs = self.transfer_batch_to_device(
            (args, kwargs), device=self.device, dataloader_idx=None
        )
        return fn(self, *args, **kwargs)

    return auto_transfer_args
