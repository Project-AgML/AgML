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

from agml.backend.tftorch import torch, tf, get_backend

def set_seed(seed = None):
    """Sets a new random seed. If None, uses a random seed."""
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    if get_backend() == 'torch':
        torch.random.manual_seed(seed)
    elif get_backend() == 'tensorflow':
        tf.random.set_seed(seed)

