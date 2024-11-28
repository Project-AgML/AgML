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
Generates the shape information file for new datasets or changed datasets.
"""

import argparse
import os
import pickle
import shutil

import numpy as np
from tqdm import tqdm

import agml

# Parse input arguments (get the datasets to re-generate).
ap = argparse.ArgumentParser()
ap.add_argument(
    "--datasets",
    nargs="+",
    required=True,
    help="The datasets you want to generate (or re-generate) shape information "
    "for. These should be in the `public_datasources.json` file, and if it is"
    "a new dataset, the folder should be in `~/.agml/datasets`. To re-generate"
    'shape information for all datasets, use the "all" command. ',
)
datasets = ap.parse_args().datasets

# Load in the original contents of the shape info file.
shape_info_file = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "agml", "_assets", "shape_info.pickle"
)
if os.path.exists(shape_info_file):
    with open(shape_info_file, "rb") as f:
        shape_contents = pickle.load(f)
else:
    shape_contents = {}

# Determine which datasets to download.
if datasets[0] == "all":
    datasets = agml.data.public_data_sources()

# Parse through and update the shape contents.
for ds in tqdm(datasets, desc="Processing Datasets"):
    if hasattr(ds, "name"):
        ds = ds.name
    current_shapes, leave = [], True
    if not os.path.exists(
        os.path.join(os.path.expanduser("~"), ".agml", "datasets", ds)
    ):
        leave = False
    loader = agml.data.AgMLDataLoader(ds)
    for contents in tqdm(loader, desc="Loader Iteration", leave=False):
        image, _ = contents
        if isinstance(image, dict):
            image = image["image"]
        current_shapes.append(image.shape)
    shape_contents[loader.name] = np.unique(current_shapes, return_counts=True, axis=0)
    if not leave:
        shutil.rmtree(loader.dataset_root)

# Save the contents once again.
with open(shape_info_file, "wb") as f:
    pickle.dump(shape_contents, f)
