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
Generates the normalization info for new or changed datasets.
"""

import argparse
import json
import os
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

# Load in the original contents of the data sources JSON.
source_info_file = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "agml",
    "_assets",
    "public_datasources.json",
)
if os.path.exists(source_info_file):
    with open(source_info_file, "r") as f:
        source_info = json.load(f)
else:
    raise ValueError(
        f"The public data source file is missing "
        f"(found nothing at {source_info_file}."
    )

# Determine which datasets to download.
if datasets[0] == "all":
    datasets = agml.data.public_data_sources()
else:
    datasets = [agml.data.source(i) for i in datasets]

# Parse through and calculate the mean/std.
for ds in tqdm(datasets, desc="Processing Datasets"):
    leave = True
    if not os.path.exists(
        os.path.join(os.path.expanduser("~"), ".agml", "datasets", ds.name)
    ):
        leave = False
    loader = agml.data.AgMLDataLoader(ds if isinstance(ds, str) else ds.name)

    # Get the mean/std of individual images.
    num_images, mean, std = 0, 0.0, 0.0
    for contents in tqdm(loader, desc="Loader Iteration", leave=False):
        num_images += 1
        image, _ = contents  # noqa
        if isinstance(image, dict):
            image = image["image"]
        image = (image / 255).astype(np.float32)
        image = np.reshape(np.transpose(image, (2, 0, 1)), (3, -1))
        mean += image.mean(-1)
        std += image.std(-1)
    mean = tuple((mean / num_images).tolist())  # noqa
    std = tuple((std / num_images).tolist())  # noqa
    source_info[loader.name]["stats"] = {"mean": tuple(mean), "std": tuple(std)}
    if not leave:
        shutil.rmtree(loader.dataset_root)

with open(source_info_file, "w") as f:
    json.dump(source_info, f, indent=4)
