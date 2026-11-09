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
Agricultural statistics, one subpackage per data source:

- `agml.ag_stats.nass`: USDA NASS crop statistics for the United States.
- `agml.ag_stats.calicropyield`: the California Crop Yield Benchmark
  (county crop yield with satellite, ET, climate, and soil layers).

Each source has its own optional dependencies, so a source is only
imported when it is first used, e.g. `agml.ag_stats.nass.NASSDataLoader()`
or `from agml.ag_stats.calicropyield import CaliCropYieldLoader`.
"""

import importlib as _importlib

# Every data source. Add a new source's subpackage name here.
SOURCES = ("nass", "calicropyield")

__all__ = list(SOURCES)


def __getattr__(name):
    if name in SOURCES:
        return _importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(SOURCES))
