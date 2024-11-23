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

import importlib
import os
import subprocess

from agml.utils.logging import log

# Only install the packages that are not already installed, and ignore
# the ones that are not needed for basic YOLO training (minimize environment).
manually_filtered_yolo_requirements = [
    "thop>=0.1.1",
    "ultralytics>=8.2.34",
    "psutil",
    "gitpython>=3.1.30",
]


def install_and_configure_ultralytics():
    """Installs and configures the Ultralytics package for YOLO11 training and inference."""

    # install the required packages
    for package in manually_filtered_yolo_requirements:
        try:
            importlib.import_module(package)
        except ImportError:
            subprocess.run(["pip", "install", package])

    # install the ultralytics package
    subprocess.run(["pip", "install", "ultralytics"])

    # ensure that packages are the right version (too high causes errors)
    pillow_version = importlib.import_module("PIL").__version__
    pillow_version_tuple = tuple(map(int, pillow_version.split(".")))
    if pillow_version_tuple > (9, 5, 0):
        log(f"Detected Pillow version ({pillow_version}) is too high, downgrading to 9.5.0")
        subprocess.run(["pip", "install", '"Pillow<=9.5.0"'])

    hf_hub_version = importlib.import_module("huggingface_hub").__version__
    hf_hub_version_tuple = tuple(map(int, hf_hub_version.split(".")))
    if hf_hub_version_tuple > (0, 24, 7):
        log(f"Detected huggingface-hub version ({hf_hub_version}) is too high, downgrading to 0.24.7")
        subprocess.run(["pip", "install", '"huggingface-hub==0.24.7"'])
