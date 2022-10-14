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

__version__ = '0.4.4'
__all__ = ['data', 'backend', 'viz']


# If AgML is being imported for the first time, then we need to setup
# the module, namely prepping the config file.
def _setup():
    import os as _os
    import json as _json
    if not _os.path.exists(_os.path.expanduser('~/.agml')):
        _os.makedirs(_os.path.expanduser('~/.agml'))
        with open(_os.path.join(
                _os.path.expanduser('~/.agml/config.json')), 'w') as f:
            _json.dump({'data_path': _os.path.expanduser('~/.agml/datasets')}, f)
_setup(); del _setup # noqa


# There are no top-level imported functions or classes, only the modules.
from . import data, backend, synthetic, viz




