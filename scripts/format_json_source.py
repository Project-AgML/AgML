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
Convenience script to reformat the public data source
JSON file in a nicer-looking format (for readability).
"""

import json
import os

SOURCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "agml",
    "_assets",
    "public_datasources.json",
)
CITATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "agml",
    "_assets",
    "source_citations.json",
)

with open(SOURCE_PATH, "r") as f:
    contents = json.load(f)
with open(SOURCE_PATH, "w") as f:
    json.dump(contents, f, indent=4)

with open(CITATION_PATH, "r") as f:
    contents = json.load(f)
with open(CITATION_PATH, "w") as f:
    json.dump(contents, f, indent=4)
