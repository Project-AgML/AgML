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

import torch
import torch.nn as nn


@torch.jit.script
def accuracy(output, target):
    """Computes the accuracy between `output` and `target`."""
    batch_size = target.size(0)
    _, pred = torch.topk(output, 1, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:1].reshape(-1).float().sum(0, keepdim = True)
    return correct_k.mul_(100.0 / batch_size)


class Accuracy(nn.Module):
    """A metric to compute accuracy for image classification tasks.

    This class is used as a wrapper around accuracy calculations, which allows for
    accumulation of predictions over time. The `update` method can be used to update
    data, then `compute` to get the calculated accuracy, and finally `reset` can be
    used to reset the accumulators to an empty state, allowing new calculations.
    """

    def __init__(self):
        # Construct the data accumulators.
        super(Accuracy, self).__init__()
        self._prediction_data, self._truth_data = [], []

    def update(self, pred_data, gt_data):
        """Updates the state of the accuracy metric.

        The `pred_data` and `gt_data` arguments should both be sequences of integer
        labels (e.g., an array of one-dimension), *not* one-hot labels. So, any
        activation operations or softmax operations must be applied before inputting
        data into the accuracy metric.
        """
        if not len(pred_data) == len(gt_data):
            raise ValueError("Predictions and truths should be the same length.")
        self._prediction_data.extend(pred_data)
        self._truth_data.extend(gt_data)

    def compute(self):
        """Computes the accuracy between the predictions and ground truths."""
        return accuracy(torch.tensor(self._prediction_data),
                        torch.tensor(self._truth_data))

    def reset(self):
        """Resets the accumulator states."""
        del self._prediction_data, self._truth_data
        self._prediction_data, self._truth_data = [], []

