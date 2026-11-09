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
This module downloads and reads the California Crop Yield Benchmark:
county crop yield for California with Cropland Data Layer, Landsat,
OpenET, irrigation, DayMet climate, and soil layers for each county,
hosted on the Hugging Face Hub.
"""

try:
    import huggingface_hub  # noqa: F401
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import rasterio  # noqa: F401
    import xarray  # noqa: F401
except ImportError:
    raise ImportError(
        "The `agml.ag_stats.calicropyield` module needs huggingface_hub, pandas, pyarrow, rasterio, "
        "and xarray. Install them with `pip install agml[calicropyield]`."
    )

from .catalog import CDL_CLASSES, COUNTIES, DEFAULT_REPO_ID, LAYERS
from .loader import CaliCropYieldLoader

__all__ = ["CaliCropYieldLoader", "CDL_CLASSES", "COUNTIES", "DEFAULT_REPO_ID", "LAYERS"]
