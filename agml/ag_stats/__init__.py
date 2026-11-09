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
This module gives access to USDA NASS crop statistics (yield, area,
production, crop progress, crop condition, and forecasts) at national,
state, district, and county level, with queries by crop, place, and year.
"""

try:
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
except ImportError:
    raise ImportError(
        "The `agml.ag_stats` module needs pandas and pyarrow. Install them with `pip install agml[ag_stats]`."
    )

from .catalog import NASSCatalog
from .loader import DEFAULT_REPO_ID, FORMATS, TABLES, NASSDataLoader
