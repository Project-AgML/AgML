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

from .classification import ClassificationModel
from .segmentation import SegmentationModel
from .detection import DetectionModel
from . import metrics
from . import losses
from . import preprocessing

