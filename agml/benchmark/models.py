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

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4
from torchvision.models.segmentation import deeplabv3_resnet50

from timm.models.efficientnet import EfficientNet


class EfficientNetB4Transfer(nn.Module):
    """Represents a transfer learning EfficientNetB4 model.

    This is the base benchmarking model for image classification, using
    the EfficientNetB4 model with two added linear fully-connected layers.
    """
    def __init__(self, num_classes, pretrained = True):
        super(EfficientNetB4Transfer, self).__init__()
        self.base = efficientnet_b4(pretrained = pretrained)
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(256, num_classes)

    def forward(self, x, **kwargs): # noqa
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x


class DeepLabV3Transfer(nn.Module):
    """Represents a transfer learning DeepLabV3 model.

    This is the base benchmarking model for semantic segmentation,
    using the DeepLabV3 model with a ResNet50 backbone.
    """
    def __init__(self, num_classes, pretrained = True):
        super(DeepLabV3Transfer, self).__init__()
        self.base = deeplabv3_resnet50(
            pretrained = pretrained,
            num_classes = num_classes
        )

    def forward(self, x, **kwargs): # noqa
        return self.base(x)



