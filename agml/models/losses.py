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

"""
A collection of useful loss functions for agricultural ML tasks.

Some of these have been put into use for benchmarking (see the
`training` directory for examples of usage in training scripts).
"""

import torch
import torch.nn as nn


def dice_loss(y_pred, y):
    """Calculates `dice loss` for semantic segmentation.

    See https://arxiv.org/abs/1707.03237 for an in-depth explanation.
    """
    # Convert ground truth to float for compatibility in operations.
    y = y.float()

    # Determine whether this is a multi-class or binary task.
    try: # Multi-class segmentation
        c, h, w = y.shape[1:]
    except: # Binary segmentation
        h, w = y.shape[1:]; c = 1 # noqa

    # Sigmoid for the outputs (since this is automatically done by binary
    # cross-entropy loss with logits, the actual base model for semantic
    # segmentation does not include any sigmoid activation in it, only the
    # `predict()` function wraps it. So, we do it here as well.
    y_pred = torch.sigmoid(y_pred)

    # Run the dice loss calculations.
    pred_flat = torch.reshape(y_pred, [-1, c * h * w])
    y_flat = torch.reshape(y, [-1, c * h * w])
    intersection = 2.0 * torch.sum(pred_flat * y_flat, dim = 1) + 1e-6
    denominator = torch.sum(pred_flat, dim = 1) \
                  + torch.sum(y_flat, dim = 1) + 1e-6
    return 1. - torch.mean(intersection / denominator)


class DiceLoss(nn.Module):
    def forward(self, x, target, **kwargs): # noqa
        return dice_loss(x, target)

