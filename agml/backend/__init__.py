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
Controls the AgML backend system.

The first part of the backend is the backend deep learning library.
The backend, one of {TensorFlow, PyTorch}, primarily exists for internal
purposes, e.g. figuring out which methods to use in the data module or
the actual model configuring/training modules.

Secondly, the backend controls the loading/saving procedure for files within
AgML, specifically data loaded or generated from the data module.
"""

from .config import (
    data_save_path,
    set_data_save_path,
    synthetic_data_save_path,
    set_synthetic_save_path,
    model_save_path,
    set_model_save_path,
    clear_all_datasets,
    downloaded_datasets
)
from .tftorch import get_backend, set_backend
from .random import set_seed
from . import experimental
