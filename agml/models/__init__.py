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
This module contains PyTorch pre-trained weights and benchmarks for
commonly used deep learning models on agricultural datasets within AgML.
"""

# Before anything can be imported, we need to run checks for PyTorch and
# PyTorch Lightning, as these are not imported on their own.
try:
    import torch
except ImportError:
    raise ImportError('Could not find a PyTorch installation. If you want to use '
                      'the models in `agml.models`, you will need to install PyTorch '
                      'first. Try `pip install torch` to do so.')

try:
    import pytorch_lightning
except ImportError:
    raise ImportError('Could not find a PyTorch Lightning installation. If you want to '
                      'use the models in `agml.models`, you will need to install PyTorch '
                      'Lightning first. Try `pip install pytorch-lightning` to do so.')


from .classification import ClassificationModel
from .segmentation import SegmentationModel
from .detection import DetectionModel
from . import metrics
from . import losses
from . import preprocessing
from .benchmarks import get_benchmark

